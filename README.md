# Build rcnn
pip install cython

pushd retinaface_detection/detection/fd/models/retinaface/

make all

popd

# Run project

make build

make up

make logs

go to 0.0.0.0:/8000

# After you finish

make down 

