# Build rcnn
pip install cython

pushd retinaface_detection/detection/fd/models/retinaface/

make all

popd

# Run project

Create an .env.dev file with the following contents:

DEBUG=1
SECRET_KEY=om+mod^k%vsvrw(^s%7g(0#utf#7m0^qcro211m&0l!im*j@3z
DJANGO_ALLOWED_HOSTS=localhost 0.0.0.0 127.0.0.1 [::1]

Then in terminal:

make build

make up

make logs

After that, go to 0.0.0.0:/8000

# After you finish

make down 

