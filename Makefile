.PHONY: dvc_data, dvc_models, dvc_commit_data

dvc: dvc_data dvc_models

dvc_data:
	dvc pull data

dvc_models:
	dvc pull models

dvc_open_data:
	dvc pull open_data

dvc_commit_data: dvc_data
	dvc commit -R data

dvc_clear_cache:
	rm -rf .dvc/cache