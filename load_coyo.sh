for i in {1..65}; do
    python -m fsconnectors download --config_path ~/kurkin/fsconnectors/config.yaml --s3_path s3://gigaeye-kandinsky-spark/datasets/text_image/coyo_700m/shards_coyo-700m_$i/shards_filtered --local_path /home/jovyan/data/coyo-700m/shards_coyo-700m_$i/shards_filtered --workers 128
done
