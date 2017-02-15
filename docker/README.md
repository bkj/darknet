### darknet Docker image

REST wrapper of `darknet` object detectors

#### Request formats

    Accepts urls or paths to images in the format
        {
            "urls" : ["http://somewhere.com/image.jpg"]
        }
    
    Returns
        [
          {
            "url": # url of image
            "model": # name of model
            "score": 0.99 # score
            "bbox": [ ... ] # coords of object
            "label": ... # name of object
          },
          ...
        ]