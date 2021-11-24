#differ SNR
python3 ./converting_loss/converting_loss.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --SNR -5 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 ./converting_loss/converting_loss.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 ./converting_loss/converting_loss.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --SNR 5 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 ./converting_loss/converting_loss.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --SNR 10 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 ./converting_loss/converting_loss.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --SNR 15 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1

#differ user num
python3 ./converting_loss/converting_loss.py --Nt 64 --Nr 4 --K 8 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 ./converting_loss/converting_loss.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 ./converting_loss/converting_loss.py --Nt 64 --Nr 4 --K 12 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 ./converting_loss/converting_loss.py --Nt 64 --Nr 4 --K 14 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 ./converting_loss/converting_loss.py --Nt 64 --Nr 4 --K 16 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1

#differ RB num
python3 ./converting_loss/converting_loss.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 1 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 ./converting_loss/converting_loss.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 2 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 ./converting_loss/converting_loss.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 3 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 ./converting_loss/converting_loss.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 4 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
python3 ./converting_loss/converting_loss.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 5 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1


