#Length Discriminative Heuristics

Reviewing IR efforts in recommendation systems.

Citation Info  & Bibtex Code | Summary  | My Take | References to follow up |
--------------|--------------|--------------|--------------|-------------------------|
[**Content-based Recommendation in Social Tagging Systems**, *Iván Cantador*](https://repositorio.uam.es/bitstream/handle/10486/665157/content-based_cantador_recsys_2010_ps.pdf?sequence=3), Cantador | Adaptations of BM25 and Vector space models are used to recommend content to Delicious and Last.fm users. Cosine similarity between BM25 value of user profile and item profile works the best. This concludes that penalization of the common tags gives more accuracy. | It's a tag based model, it does not uses textual data. They could have experimented with some changes in the length normalization part. Only MAP and Precision in a tabular format is shown. Parameter sensitivity graphs could make this paper more cool. | [This really cool survey](http://web.stanford.edu/class/ee378b/papers/adomavicius-recsys.pdf) |
[**Friend Recommendation Method using Physical and Social Context**, *Kwon* ](http://paper.ijcsns.org/07_book/201011/20101118.pdf), Kwon | BM25 is used to balance between the "logging in" variables vs. other friends. The score is then treated as a graphical proximity between users. | Worst paper I have ever read. How do such papers get accepted? | Nope |
[**Toward the Next Generation of Recommender Systems: A Survey of the State-of-the-Art and Possible Extensions**, *Gediminas*](http://web.stanford.edu/class/ee378b/papers/adomavicius-recsys.pdf), nextgen | Defines usage of content based recommendations using text retreival models. IR systems cannot distinguish between a well-written article and a badly written one, if they happen to use the same terms. Diversity problems too. | "This cool survey" turns out to be a hangout of teenage girls speaking buzzwords. Very biased paper. Only TF-IDF model is included while taking about information retrieval. I wished they stressed about importance of IDF and length normalization.| Usage of threshold setting in adaptive systems. |
[**Full Text Search Engine as Scalable k-Nearest Neighbor Recommendation System**, *Suchal*, Suchal](https://hal.archives-ouvertes.fr/file/index/docid/1054596/filename/wcc-final.pdf) | Very simple idea : Concatenate user's picks into a single query and using that to evaluate the top-k recommendations |No effort done for accommodating long queries. This breaks my heart. Lacks algorithmic details. Parameter sensitivity analysis is really well done. But scalability come later and accuracy comes first. | None |
[**Course Recommendation by Improving BM25 to Identity Students’ Different Levels of Interests in Courses**, *Xin Wang*, course](http://info.cic.tsinghua.edu.cn/upload_file/_temp/1268811498049/1268811498049.pdf) | Similar to the tag system recommender. | Irrelevant Literature Survey, Poor English, Not used standard evaluation metric, No logic on their construction of their ranking function. | Sorry, no |
