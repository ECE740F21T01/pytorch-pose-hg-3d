# Run it under root directory
mkdir data/flic
mkdir data/flic/annot
cp /data/keith/datasets/FLIC-full/examples.mat data/flic/annot/examples.mat
ln -s /data/keith/datasets/FLIC-full/images data/flic
python src/main.py --exp_id flic --dataset flic