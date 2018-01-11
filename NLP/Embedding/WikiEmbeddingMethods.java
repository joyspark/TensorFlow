package word2vec.wiki;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class WikiEmbeddingMethods {

public WikiEmbeddingMethods() {
		
	}
	
	/**
	 * 단어 사전을 맵으로 만들어주는 함수
	 * 
	 * @param map
	 * @param keyIdx
	 * @param filePath
	 * @return sucess 0, fail -1
	 */
	public long fileToMap(Map map, boolean keyIdx, String filePath) {
		
		try {
			File csv = new File(filePath);
			BufferedReader br = new BufferedReader(new FileReader(csv));
			String line = "";
			String[] words = null;
			
			while((line = br.readLine())!=null) {
				words = line.split(",");
				for(int i=0; i<words.length; i++) {
					if(keyIdx) // <인덱스, 단어>
						map.put(i, words[i]);
					else       // <단어, 인덱스>
						map.put(words[i], i);
				}
			}
			
		}catch(Exception e) {
			e.printStackTrace();
			return -1;
		}
		
		return 0;
	}
	
	/**
	 * index 데이터를 해당 단어로 바꿔주는 함수
	 * 
	 * @param dicMap
	 * @param idxs
	 * @return
	 */
	public String[] toWordDic(Map<Integer, String> dicMap, int[] idxs) {
		ArrayList<String> wordsList = new ArrayList<String>();
		
		for(int idx : idxs) {
			String w = dicMap.get(idx);
			wordsList.add(w);
		}
		
		String[] wordsArr = new String[wordsList.size()];
		wordsArr = wordsList.toArray(wordsArr);
		
		return wordsArr;
	}
	
	/**
	 * 단어를 해당 index로 바꿔주는 함수
	 * 
	 * @param dicMap
	 * @param words
	 * @return
	 */
	public Integer[] toIdxDic(Map<String, Integer> dicMap, String[] words) {
		ArrayList<Integer> idxsList = new ArrayList<Integer>();
		
		for(String word : words) {
			int idx = dicMap.get(word);
			idxsList.add(idx);
		}
		
		Integer[] idxsArr = new Integer[idxsList.size()];
		idxsArr = idxsList.toArray(idxsArr);
		
		return idxsArr;
	}
	
	/**
	 * 임베딩 벡터 읽어오는 함수
	 * 
	 * @param modelPath
	 * @param testIdxs
	 * @param topK
	 * @param dicSize
	 */
	public void getEmbedding(String modelPath, Integer[] testIdxs, int topK, int dicSize) {
		
		int dataSize = testIdxs.length;
		System.out.println("dataSize : " + dataSize);
		
		try(SavedModelBundle b = SavedModelBundle.load(modelPath, "serve")){
			
			//create a session from the Bundle
			Session sess = b.session();
			for(int i=0; i<testIdxs.length; i++) {
				Tensor x = Tensor.create(testIdxs);
				
				//run the model and get the result
				int[][] sim = sess.runner()
						.feed("valid_dataset", x)
						.fetch("similarity")
						.run()
						.get(0)
						.copyTo(new int[dataSize][dicSize]);
				/*
				System.out.println("before sort : " + sim[0][0]+", "+sim[0][1]+", "+sim[0][2]+", "+sim[0][3]);
				Arrays.sort(sim[0]);
				System.out.println("after sort : " + sim[0][0]+", "+sim[0][1]+", "+sim[0][2]+", "+sim[0][3]);
				*/
			}
		}
	}
	
	/**
	 * 프로그램 종료 메소드
	 * @param result
	 * @param msg
	 */
	public void terminated(long result, String msg) {
		if(result >-1)
			return;
		System.out.println("program terminated");
		System.exit(1);
	}
}
