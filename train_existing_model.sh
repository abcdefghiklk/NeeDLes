project_name=$1
epoch=$2
cd code
#java -jar OracleGenerator_config.jar --config ../config/NeeDLes_${project_name}.ini
/usr/local/python27/bin/python train_on_existing_model.py -config ../config/NeeDLes_${project_name}.ini -n ${epoch}
