project_name=$1
cd code
#java -jar OracleGenerator_config.jar --config ../config/NeeDLes_${project_name}.ini
/usr/local/python27/bin/python config_parser_2.py -config ../config/NeeDLes_${project_name}.ini
