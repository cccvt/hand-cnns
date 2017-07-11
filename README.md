All datasets should be in the data folder.

For instance, for UCI-EGO, it should go into  data/UCI-EGO


# Remarks

## Notes to self

Consider using a symbolic link if folder is already somewhere on the computer

`ln -s path/to/UCI-EGO data/UCI-EGO`

## Datasets

### GUN dataset

For ease of processing, I aggregated all the images into a unique folder that I name gun_all.

For this, I did

```shell
mkdir gun_all
cd gun_all
cp -R path/to/orginal/gun/*/*.png .
cp -R path/to/orginal/gun/*/*.jpg .
```

The original GUN dataset can be downloaded from [here](http://www.gregrogez.net/research/egovision4health/gun-71/)

