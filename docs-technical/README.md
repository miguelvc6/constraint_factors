# Technical Documentation

Technical docs describe repository behavior, artifact contracts, commands, and
developer-facing operations.

- [Scientific integrity remediation](09_scientific_integrity_remediation.md): implemented audit fixes, invalidated artifacts, mandatory reruns, commands, and acceptance gate.
- [Temporary paper-artifact rerun commands](TEMP_paper_artifact_rerun_after_order_fix.md): post-order-fix rebuild commands; remove after the successful rerun is recorded in the paper ledger.
- [Model and evaluation matrix](00_models_and_evaluation_matrix.md): canonical paper-facing systems and metric surface.
- [Training and evaluation plan](00_training_and_evaluation_execution_plan.md): detailed experiment order and diagnostics.
- [Model config reference](00_model_config_reference.md): supported model/training configuration fields.
- [Data downloader](01_data_downloader.md)
- [Dataframe builder](02_dataframe_builder.md): preserved splits and identity/feature vocabularies.
- [Exact stratified sampler](02b_stratified_benchmark_sampler.md)
- [Constraint registry](03_constraint_registry.md)
- [Wikidata retriever](04_wikidata_retriever.md)
- [Constraint labeler](05_constraint_labeler.md): semantic labels and primary eligibility.
- [Data specification](05_data_spec.md)
- [Graph construction](06_graph.md): identity-keyed schema-v3 PyG artifacts, complete lineage, validation, and safe pruning.
- [Training](07_train.md): proposal training and run provenance.

Research rationale and paper claims belong in the
[conceptual documentation](../docs-conceptual/README.md).
