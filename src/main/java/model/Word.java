/**************************************************************	*/
/*														    	*/
/* The source code for this program is submitted to EMNLP 2018	*/
/* with paper "Limbic: Author-Based Sentiment Aspect Modeling   */
/* Regularized with Word Embeddings and Discourse Relations"	*/
/*														    	*/
/************************************************************** */

package model;

import java.util.ArrayList;

public class Word {
	private int termIdx = -1;
	private int sentiment = -1;
	private int aspect = -1;
	private int sentiLex = -1;
	
	private ArrayList<Integer> promotionList;
	
	public int getTermIdx() {
		return termIdx;
	}

	public void setTermIdx(int termIdx) {
		this.termIdx = termIdx;
	}
	
	public int getSentiment() {
		return sentiment;
	}
	
	public void setSentiment(int sentiment) {
		this.sentiment = sentiment;
	}
	
	public int getAspect() {
		return aspect;
	}
	
	public void setAspect(int aspect) {
		this.aspect = aspect;
	}

	public int getSentiLex() {
		return sentiLex;
	}

	public void setSentiLex(int sentiLex) {
		if(this.sentiLex != -1) 
			System.out.print("Error: sentiLex is not -1");
		this.sentiLex = sentiLex;
	}

	public ArrayList<Integer> getPromotionList() {
		return promotionList;
	}

	public void setPromotionList(ArrayList<Integer> promotionList) {
		this.promotionList = promotionList;
	}


}
