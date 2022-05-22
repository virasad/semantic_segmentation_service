# Semantic Segmentation Service
Train and inference semantic segmentation models easy as ABC.


# How to use this
Set the variables in the `docker-compose.yml` file.
## Train variables:
- RESPONSE_URL: the url to send the training result
- LOGGER_URL: the url to send the training log
- IS_LOGGER_ON: enable logging or not

## Inference variables:
- WEIGHTS_DIR: the directory to read weights from it.

### Run
```
docker compose up
```
**For more details about the training please refer to the [README.md](train/readme.md) file.**

**For more details about the inference please refer to the [README.md](inference/readme.md) file.**
![Docker Compose CI](https://github.com/virasad/semantic_segmentation_service/actions/workflows/docker-image.yml/badge.svg)
