/**************************************************************	*/
/*														    	*/
/* The source code for this program is submitted to EMNLP 2018	*/
/* with paper "Limbic: Author-Based Sentiment Aspect Modeling   */
/* Regularized with Word Embeddings and Discourse Relations"	*/
/*														    	*/
/************************************************************** */

package model;

public class SemanticPair {
	private int wordIdx;
	private double similarity;
	private double posSimilarity;
	private double negSimilarity;
	
	public SemanticPair(int wordIdx, double similarity, double posSimilarity, double negSimilarity) {
		this.wordIdx = wordIdx;
		this.similarity = similarity;
		this.posSimilarity = posSimilarity;
		this.negSimilarity = negSimilarity;
	}
	
	public int getWordIdx() {
		return wordIdx;
	}
	
	public void setWordIdx(int wordIdx) {
		this.wordIdx = wordIdx;
	}
	
	public double getSimilarity() {
		return similarity;
	}
	
	public void setSimilarity(double similarity) {
		this.similarity = similarity;
	}

	public double getPosSimilarity() {
		return posSimilarity;
	}

	public void setPosSimilarity(double posSimilarity) {
		this.posSimilarity = posSimilarity;
	}

	public double getNegSimilarity() {
		return negSimilarity;
	}

	public void setNegSimilarity(double negSimilarity) {
		this.negSimilarity = negSimilarity;
	}
}
