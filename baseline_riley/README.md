How to run the experiment (on Euler):

```
bash prepare-iwslt17.sh
sbatch -o train.log --gpus=1 --mem-per-cpu=8000 train.sh

```


