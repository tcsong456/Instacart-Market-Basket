if command -v kaggle > /dev/null;then
  echo 'kaggle is installed'
else
  pip install kaggle
fi

mkdir -p ~/.kaggle data/
if [ ! -f ~/.kaggle/kaggle.json ];then
  opt=$(find . -name 'kaggle.json')
  if [ -z "$opt" ];then
    echo 'download kaggle.json file from the settings of your profile on Kaggle website and run this bash shell again'
  else
    mv $opt ~/.kaggle/kaggle.json
  fi
fi
kaggle competitions download -c instacart-market-basket-analysis --path data/  
data=$(ls data/)
unzip data/$data -d data
rm data/$data
find data/ -name '*.zip' -exec unzip -o {} -x "__MACOSX/*" -d data \; -delete 
mkdir -p metadata/ data/tmp/
