#!/bin/sh


read -p "All files will consist of around 40 gb of data, do you wish to continue? [Y/n]" -n 1 -r
echo    # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi


mkdir -p indexes

# Lucene index: indexes/lucene-index-dbpedia_annotated_full.tar.gz
wget -O indexes/lucene-index-dbpedia_annotated_full.tar.gz https://surfdrive.surf.nl/files/index.php/s/ItjlwVhm8sApcZS/download
tar -xvf indexes/lucene-index-dbpedia_annotated_full.tar.gz -C /indexes

mkdir -p resources/wikipedia2vec/wikipedia-20190701
# wikipedia2vec_500.pkl.tar.gz
wget -O resources/wikipedia2vec/wikipedia-20190701/wikipedia2vec_500.pkl.tar.gz https://surfdrive.surf.nl/files/index.php/s/mOYK4gZfI3yjsZd/download
tar -xvf resources/wikipedia2vec/wikipedia-20190701/wikipedia2vec_500.pkl.tar.gz -C /resources/wikipedia2vec/wikipedia-20190701





mkdir -p output
# monobert-large-msmarco-dbpedia_acc_batch_64_e6_annotated.tar.gz
wget -O output/monobert-large-msmarco-dbpedia_acc_batch_64_e6_annotated.tar.gz https://surfdrive.surf.nl/files/index.php/s/gfCY1dc5CdkbS5S/download
tar -xvf output/monobert-large-msmarco-dbpedia_acc_batch_64_e6_annotated.tar.gz -C /output


# monobert-large-msmarco-dbpedia_acc_batch_64_e6_noent.tar.gz
wget -O output/monobert-large-msmarco-dbpedia_acc_batch_64_e6_noent.tar.gz https://surfdrive.surf.nl/files/index.php/s/5KQIRtiKikObJDG/download
tar -xvf output/monobert-large-msmarco-dbpedia_acc_batch_64_e6_noent.tar.gz -C /output


wget -O output/monobert-large-msmarco-finetuned_acc_batch_testmodel_acc_batch_600k_64_e6.tar.gz  https://surfdrive.surf.nl/files/index.php/s/eJsvZLceqi6kPeY
tar -xvf output/monobert-large-msmarco-finetuned_acc_batch_testmodel_acc_batch_600k_64_e6.tar.gz  -C /output

mkdir -p data

wget -O data/DBpedia-Entity.tar.gz https://surfdrive.surf.nl/files/index.php/s/gUwRAg6XQbTgx91/download
tar -xvf data/DBpedia-Entity.tar.gz /data

wget -O data/DBpedia-Entity-noent.tar.gz https://surfdrive.surf.nl/files/index.php/s/EJbDiEwqOYglEgD/download
tar -xvf data/DBpedia-Entity-noent.tar.gz /data
