
GUIDE: Evaluating ASTE Model on cameraCOQE Dataset
===================================================

Your cameraCOQE data has been converted to ASTE format:
  Location: data/cameraCOQE_ASTE/
  Files: train_aste.json, dev_aste.json, test_aste.json
  Format: [{"sentence": "...", "triplets": [{"aspect_term": "...", "opinion_term": "...", "aspect_sentiment_label": "..."}, ...]}, ...]

DATA STATISTICS:
  Train: 649 sentences, 2,032 triplets (3.1 triplets/sentence)
  Dev: 165 sentences, 514 triplets (3.1 triplets/sentence)
  Test: 200 sentences, 488 triplets (2.4 triplets/sentence)

CLASS DISTRIBUTION:
  The dataset is based on COMPARATIVE OPINIONS - expect different distribution
  than typical ASTE datasets which focus on single opinions

USAGE:
  1. Load cameraCOQE_ASTE data instead of original ASTE files
  2. Update your data loader to use the new file paths
  3. Run evaluation using your existing evaluation pipeline
  4. Compare results: cameraCOQE (cross-domain) vs original (in-domain)

EXPECTED PERFORMANCE DROP:
  - Cross-domain evaluation typically shows 5-15% F1 drop
  - Different domain (cameras → cameras but different annotation style)
  - Comparative opinions may be harder than standard opinions

NEXT STEPS:
  1. Update Parameter.py to reference cameraCOQE_ASTE if needed
  2. Run evaluation script with cameraCOQE data
  3. Analyze cross-domain performance
  4. Consider domain adaptation techniques
