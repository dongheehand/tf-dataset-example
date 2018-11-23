## Example code for using tensorflow dataset API
Tensorflow dataset API enables you to build complex input pipelines from simple, reusable pieces.

The example code is for loading image files

#### without TFRecords file
If all of your input image files fit in memory,
```
python data_loader.py --in_memory True
```

If all of your input image files do not fit in memory,
```
python data_loader.py --in_memory False
```
#### using TFRecords file
First of all, you should generate TFRecord files
```
python make_tf_records.py
```
You can process TFRecord files!

```
python data_loader.py --tf_records True --data_direc ./tf_records
```

### Comments
If you have any questions or comments on my codes, please email to me. [son1113@snu.ac.kr](mailto:son1113@snu.ac.kr)