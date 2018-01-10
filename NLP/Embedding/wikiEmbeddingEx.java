package co.kr.saramin.word2vec.wiki;

import java.util.HashMap;
import java.util.Map;

import co.kr.saramin.word2vec.wiki.WikiEmbeddingMethods;

public class WikiEmbeddingEx {

	static WikiEmbeddingMethods mthods = new WikiEmbeddingMethods();
	
	 public static void main(String[] args) throws Exception {
		 long startTime = System.currentTimeMillis();
		 
		 String modelPath = "model/embedding_java_2/";
		 String dataPath= "data/";
		 
		 // 사전 : <단어,인덱스> 맵
		 Map<String, Integer> wordIdMap = new HashMap<String, Integer>();
		 mthods.terminated(mthods.fileToMap(wordIdMap, false, dataPath+"ordered_words.csv"), "fileToMap : wordIdMap");
		 // 사전 : <인덱스, 단어> 맵
		 Map<String, Integer> idWordMap = new HashMap<String, Integer>();
		 mthods.terminated(mthods.fileToMap(idWordMap, true, dataPath+"ordered_words.csv"),"fileToMap : idWordMap");
		 
		 int dicSize = wordIdMap.size();
		 //System.out.println("Dic size : "+ dicSize);
		 //확인 출력
		 /*int i = 0;
		 for(String word : wordIdMap.keySet()) {
			 int idx = wordIdMap.get(word);
			 System.out.print(word +" : "+idx);
			 System.out.println(" ==> " + idWordMap.get(idx));
			 i++;
			 if(i == 10)
				 break;
		 }
		 System.out.println("UNK : 0  ==> " + idWordMap.get(0));
		 
		 System.out.println("wordIdMap size : "+wordIdMap.size());
		 System.out.println("idWordMap size : "+idWordMap.size());*/
		 		 
		 // 테스트 데이터
		 String[] testWords = {"일","성장","학교","성실"};
		 Integer[] testIdxs = mthods.toIdxDic(wordIdMap, testWords);
		 
		 //확인 출력
		 /*int i=0;
		 for(int idx : testIdxs) {
			 System.out.println(testWords[i]+" = "+idx+" ==> "+idWordMap.get(idx));
			 i++;
		 }
		 System.out.println("i : "+i);
		 System.out.println("testWords size : "+testWords.length);*/

		 // 뽑아낼 테스트 데이터의 최 단거리 단어 수
		 int topK = 5;
		 
		 // 임베딩 벡터 사용하기
		 mthods.getEmbedding(modelPath, testIdxs, topK, dicSize); //==> 여기서 에러남..python 부분이랑 비교해가며 해결해야 할 것 같음(20180105)
		 
		 long endTime = System.currentTimeMillis();
		 System.out.println("Elapsed Time : " + (endTime-startTime)/1000 + "s");
	 }	 
	 
}
