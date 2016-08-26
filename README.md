#Length Discriminative Heuristics

Reviewing IR efforts in recommendation systems.

Citation Info  & Bibtex Code | Summary  | My Take | References to follow up |
--------------|--------------|--------------|--------------|-------------------------|
[**Content-based Recommendation in Social Tagging Systems**, *Iván Cantador*](https://repositorio.uam.es/bitstream/handle/10486/665157/content-based_cantador_recsys_2010_ps.pdf?sequence=3), Cantador | Adaptations of BM25 and Vector space models are used to recommend content to Delicious and Last.fm users. Cosine similarity between BM25 value of user profile and item profile works the best. This concludes that penalization of the common tags gives more accuracy. | It's a tag based model, it does not uses textual data. They could have experimented with some changes in the length normalization part. Only MAP and Precision in a tabular format is shown. Parameter sensitivity graphs could make this paper more cool. | [This really cool survey](http://web.stanford.edu/class/ee378b/papers/adomavicius-recsys.pdf) |
[**Friend Recommendation Method using Physical and Social Context**, *Kwon* ](http://paper.ijcsns.org/07_book/201011/20101118.pdf), Kwon | BM25 is used to balance between the "logging in" variables vs. other friends. The score is then treated as a graphical proximity between users. | Worst paper I have ever read. How do such papers get accepted? | Nope |
[**Toward the Next Generation of Recommender Systems: A Survey of the State-of-the-Art and Possible Extensions**, *Gediminas*](http://web.stanford.edu/class/ee378b/papers/adomavicius-recsys.pdf), nextgen | Defines usage of content based recommendations using text retrieval models. IR systems cannot distinguish between a well-written article and a badly written one, if they happen to use the same terms. Diversity problems too. | "This cool survey" turns out to be a hangout of teenage girls speaking buzzwords. Very biased paper. Only TF-IDF model is included while taking about information retrieval. I wished they stressed about importance of IDF and length normalization.| Usage of threshold setting in adaptive systems. |
[**Full Text Search Engine as Scalable k-Nearest Neighbor Recommendation System**, *Suchal*, Suchal](https://hal.archives-ouvertes.fr/file/index/docid/1054596/filename/wcc-final.pdf) | Very simple idea : Concatenate user's picks into a single query and using that to evaluate the top-k recommendations |No effort done for accommodating long queries. This breaks my heart. Lacks algorithmic details. Parameter sensitivity analysis is really well done. But scalability come later and accuracy comes first. | None |
[**Course Recommendation by Improving BM25 to Identity Students’ Different Levels of Interests in Courses**, *Xin Wang*](http://info.cic.tsinghua.edu.cn/upload_file/_temp/1268811498049/1268811498049.pdf), course | Similar to the tag system recommender. | Irrelevant Literature Survey, poor English, not used standard evaluation metric, no logic given on their construction of their ranking function. | Sorry, no |
[**Mining the Real-Time Web: A Novel Approach to Product Recommendation**, *Esparza*](http://researchrepository.ucd.ie/bitstream/handle/10197/3746/kbs-2010-revised%20copy.pdf?sequence=1), Esparza | Tags and reviews of movies are served as an TF-IDF component. Reviews tend to have better performance than tags because they provide some noise and undiscovered information which then results high IDF values. Surprising to see that this works better than collaborative filtering. They used precision-recall curves and F1 scores for evaluations. In some places BM25 underperformed than TF-IDF (I know why, authors don't) | Very crucial paper. I wished the lit survey was focused on recommender systems using text retrieval functions. Good idea to compare different datasets. No MAP and NDCG is used for evaluation. No logic of choosing hyperparameters is given. | TO-DO |
[**A Study of Heterogeneity in Recommendations for a Social Music Service**, *Alejandro*](http://arantxa.ii.uam.es/~cantador/doc/2010/hetrec10.pdf), Bellogin | Refer to Cantador one | Similar paper to that of Cantador dude. However, it introduces more performance metrics. Collaborative Filtering have more novel and diverse recommendations. IR has more coverage and better accuracy. TF-IDF yields better results because they didn't tune BM25. | TO- DO |
[**Profiling vs. Time vs. Content: What does Matter for Top-k Publication Recommendation based on Twitter Profiles?**, *Chifumi*](http://arxiv.org/pdf/1603.07016.pdf), twitter | Building up a publication recommendation system. They used CF-IDF, which is basically giving more importance to certain terms. HCF-IDF is a fancy way to give weight to more prominent terms. CF-IDF with Sliding window concept yielded best results. LDA had an underwhelming performance. Used plenty of evaluation metrics to evaluate their models. | Question to ask myself : Can my research work club the similarity score / sliding window metrics and profiling methods? Authors were more interested in differences between the algorithms rather than behaviour of the algos. | - |
[**A Query-oriented Approach for Relevance in Citation Networks**, *Totti*](http://www2016.net/proceedings/companion/p401.pdf), Totti | IQRA-TC uses citation recommendation graph. The weights are : Citation context (similarity between two papers), Query Similarity (similarity between query and article) and Age Decay (to penalize older articles). IQRA-ML uses approach similar to pagerank metrics. Publications, authors, venues and keywords are taken into account. TF-IDF score is used to calculate the pagrank metric. IQRA-TC works out the best. | They haven't tuned their baselines. Methodology of the choosing hyperparameters is not clear. Parameter sensitivity lacks for IQRA-TC. NDCG and MAP is used for evaluation for models. Although this paper raises an important point that different users want different search techniques, but their justification kinda leads to overfitting. | - |


Reviewing the IR Models

Model name | Link | Description |
-----------|------|-------------|
Pivoted Normalization | [Singhal et al.](http://dl.acm.org/citation.cfm?id=243206&dl=ACM&coll=DL&CFID=830327898&CFTOKEN=13998328) | - |
Okapi BM25 | [Robertson et al.](http://dl.acm.org/citation.cfm?id=188561) | - |
Dirchlet Prior | [Zhai et al.](http://dl.acm.org/citation.cfm?id=984322) | - |
PL2 | [Amati et al.](http://dl.acm.org/citation.cfm?id=582416) | - |
