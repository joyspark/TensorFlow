import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

public class LoadPythonModel {

	static int ROW = 0;
	static int FEATURE = 0;
	
	public static void main(String[] args) throws IOException{
		System.out.println("TensorFlow version : "+TensorFlow.version());
		
		String filePath = "./data/test.csv";
		
		//get shape of data
		getDataSize(filePath);
		System.out.print("[number of row] ==> "+ ROW);
		System.out.println(" / [number of feature] ==> "+ FEATURE);
		float[][] testInput = new float[ROW][FEATURE];
		
		//insert csv data to matrix
		csvToMtrx(filePath, testInput);
		printMatrix(testInput);
		
		//load the model bundle
		try(SavedModelBundle b = SavedModelBundle.load("tmp/fromPython", "serve")){
			
			//create a session from the Bundle
			Session sess = b.session();
			
			//create an input Tensor
			Tensor x = Tensor.create(testInput);
			
			//run the model and get the result
			float[][] y = sess.runner()
					.feed("x", x)
					.fetch("h")
					.run()
					.get(0)
					.copyTo(new float[ROW][1]);
			
			//print out the result
			for(int i=0; i<y.length;i++)
				System.out.println(y[i][0]);
		}
	}
	
	/**
	 * csv 파일의 행/열 사이즈 측정
	 * @param filePath
	 * @throws IOException
	 */
	public static void getDataSize(String filePath) throws IOException {
		try {
			//read csv data file
			File csv = new File(filePath);
			BufferedReader br = new BufferedReader(new FileReader(csv));
			String line = "";
			String[] field = null;
			
			while((line=br.readLine())!=null) {
				field = line.split(",");
				ROW++;
			}
			
			FEATURE = field.length;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * csv 파일 데이터를 행렬로 옮김
	 * @param filePath
	 * @param mtrx
	 * @throws IOException
	 */
	public static void csvToMtrx(String filePath, float[][] mtrx) throws IOException {
		try {
			//read csv data file
			File csv = new File(filePath);
			BufferedReader br = new BufferedReader(new FileReader(csv));
			String line = "";
			String[] field = null;
			
			for(int i=0; i<mtrx.length; i++) {
				if((line=br.readLine())!= null) {
					field = line.split(",");
					for(int j=0; j<field.length; j++) {
						mtrx[i][j] = Float.parseFloat(field[j]);
					}
				}
			}
		}catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	/**
	 * 행렬 값 확인용 출력
	 * @param mtrx
	 */
	public static void printMatrix(float[][] mtrx) {
		System.out.println("============ARRAY VALUES============");
		for(int i=0; i<mtrx.length; i++) {
			if(i==0)
				System.out.print("[");
			else
				System.out.println();
			for(int j =0; j<mtrx[i].length; j++) {
				System.out.print("["+mtrx[i][j]+"]");
			}
		}
		System.out.println("]");
	}
}
