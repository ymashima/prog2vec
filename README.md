# struc2vec(開発中)
struc2vecはソースコード集合を，抽象構文木の部分構造（経路ベース）に基づいて数値表現に変換するツールです．

<center><img width="70%" src="https://github.com/ymashima/struc2vec/raw/develop/images/struc2vec.png" /></center>

# Requirements
以下のツールをシステム内部に使用しています．

* [svm-perf v.3.00](http://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html)
* [elasticsearch v.5.6.13](https://www.elastic.co/jp/downloads/past-releases/elasticsearch-5-6-13)
* [clang-format LLVM v.7.0.1](http://releases.llvm.org/download.html)

## svm-perf
svm-perfはJoachimsらによって開発され，support vector machine(svm)のカーネル関数をlinearに限定した際に高速に学習するツールです．

```
@inproceedings{joachims2006training,
  title={Training linear SVMs in linear time},
  author={Joachims, Thorsten},
  booktitle={Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={217--226},
  year={2006},
  organization={ACM}
}
```

# Example dataset
exampleフォルダにあるデータセットは， Allamanisらがgithubより収集した[データセット](http://groups.inf.ed.ac.uk/cup/javaGithub/)の一部を使用しています．

```
@inproceedings{githubCorpus2013,
	author={Allamanis, Miltiadis and Sutton, Charles},
	title={{Mining Source Code Repositories at Massive Scale using Language Modeling}},
	booktitle={The 10th Working Conference on Mining Software Repositories},	
	year={2013},
	pages={207--216},
	organization={IEEE}
}
```

# Running
hoge

