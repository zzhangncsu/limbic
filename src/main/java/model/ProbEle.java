/**************************************************************	*/
/*														    	*/
/* The source code for this program is submitted to EMNLP 2018	*/
/* with paper "Limbic: Author-Based Sentiment Aspect Modeling   */
/* Regularized with Word Embeddings and Discourse Relations"	*/
/*														    	*/
/************************************************************** */

package model;

public class ProbEle implements Comparable<ProbEle> {
	private int idx = -1;
	private double prob;
	private int sentiment = -1;
	private int aspect = -1;
	
	public ProbEle(int idx, double prob) {
		this.idx = idx;
		this.prob = prob;
	}
	
	public ProbEle(int sent, int aspect, double prob) {
		this.sentiment = sent;
		this.aspect = aspect;
		this.prob = prob;
	}
	
	public int getIdx() {
		return idx;
	}
	
	public void setIdx(int idx) {
		this.idx = idx;
	}
	
	public double getProb() {
		return prob;
	}
	
	public void setProb(double prob) {
		this.prob = prob;
	}

	@Override
	public int compareTo(ProbEle t) {
		return prob > t.getProb() ? -1 : prob < t.getProb() ? 1 : 0;
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
	
}
