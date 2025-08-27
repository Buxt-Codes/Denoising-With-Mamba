## Training Data

Process Location Metadata with Reviews
1. Ensure each review for the locations are all relevant
2. Strip out the reviews in which are highly targetted to a specific category or store
    - Link these reviews to a specific store/category
    - Use them as negatives for other stores/categories

I want a dataset with ONLY positives that can act as ground truths.
I want another dataset with Ads to inject as negatives.
I want another dataset with SPECIFIC positives that can be seen as irrelevant if not for the same location/category.
I want another dataset with reviews from people who has not visited the place as well.

4 Datasets, one for positives, three for negatives.