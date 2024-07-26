
source /home/ec2-user/ai/bin/activate

cd /home/ec2-user/ai/ai-model/src
uvicorn main:app --host 0.0.0.0 --port 8000