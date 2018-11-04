/**************************************************************	*/
/*														    	*/
/* The source code for this program is submitted to EMNLP 2018	*/
/* with paper "Limbic: Author-Based Sentiment Aspect Modeling   */
/* Regularized with Word Embeddings and Discourse Relations"	*/
/*														    	*/
/************************************************************** */

package model;

import java.util.ArrayList;
import java.util.HashSet;

public class Sentence {
	private int idx = -1;
	private int oriIdx = -1;
	private String rIdx = "";
	private int sentiment = -1;
	private int aspect = -1;
	private ArrayList<Word> wordList = null;
	private long timestamp = 0l;
	private double fractionalTimestamp = 0d;
	private boolean containTrans = false;
	private boolean containAddition = false;
	private boolean containBasicElement = false;
	
	public Sentence(int senti, int aspect, ArrayList<Word> words) {
		this.sentiment = senti;
		this.aspect = aspect;
		this.wordList = words;
	}
	
	  public Sentence(int senti, int aspect, ArrayList<Word> words, long timestamp) {
	      this.sentiment = senti;
	      this.aspect = aspect;
	      this.wordList = words;
	      this.timestamp = timestamp;
	   }
	
	public Sentence(int idx, int oriIdx, String rIdx, int senti, int aspect, ArrayList<Word> words) {
		this.idx = idx;
		this.oriIdx = oriIdx;
		this.setrIdx(rIdx);
		this.sentiment = senti;
		this.aspect = aspect;
		this.wordList = words;
	}
	
	public int getIdx() {
		return idx;
	}

	public void setIdx(int idx) {
		this.idx = idx;
	}

	public String getrIdx() {
		return rIdx;
	}

	public void setrIdx(String rIdx) {
		this.rIdx = rIdx;
	}
	
	public HashSet<Word> getWordSet() {
		HashSet<Word> wordSet = new HashSet<>();
		for(Word w : wordList) {
			if(!wordSet.contains(w)) {
				wordSet.add(w);
			}
		}
		return wordSet;
	}
	
	public int getOriIdx() {
		return oriIdx;
	}

	public void setOriIdx(int oriIdx) {
		this.oriIdx = oriIdx;
	}

	public int getWordCount(int wordTermIdx) {
		int count = 0;
		for(Word w : wordList) {
			if(w.getTermIdx() == wordTermIdx) {
				count ++;
			}
		}
		return count;
	}
	
	public int getNumWords() {
		if(this.wordList != null) {
			return this.wordList.size();
		} else {
			return -1;
		}
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
	
	public ArrayList<Word> getWordList() {
		return wordList;
	}
	
	public void setWordList(ArrayList<Word> wordList) {
		this.wordList = wordList;
	}

   public long getTimestamp() {
      return timestamp;
   }

   public void setTimestamp(long timestamp) {
      this.timestamp = timestamp;
   }

   public double getFractionalTimestamp() {
      return fractionalTimestamp;
   }

   public void setFractionalTimestamp(double fractionalTimestamp) {
      this.fractionalTimestamp = fractionalTimestamp;
   }

public boolean isContainTrans() {
	return containTrans;
}

public void setContainTrans(boolean containTrans) {
	this.containTrans = containTrans;
}

public boolean isContainAddition() {
	return containAddition;
}

public void setContainAddition(boolean containAddition) {
	this.containAddition = containAddition;
}

public boolean isContainBasicElement() {
	return containBasicElement;
}

public void setContainBasicElement(boolean containBasicElement) {
	this.containBasicElement = containBasicElement;
}
	
}
