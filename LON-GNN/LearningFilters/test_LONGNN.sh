python training.py --filter_type low --net StdJacobiSGNNS --jacob_a -0.8 --jacob_b 0.9 --alpha 1.0 --lr 0.05 --wd 5e-05 --lr_ab 0.0002 > test_on_low.out 2>test_on_low.err
python training.py --filter_type band --net StdJacobiSGNNS --jacob_a -0.95 --jacob_b 0.95 --lr 0.05 --wd 5e-05 --alpha 0.75 --lr_ab 0.0002 > test_on_band.out 2> test_on_band.err
python training.py --filter_type comb --net StdJacobiSGNNS --jacob_a -0.9 --jacob_b 0.75 --lr 0.05 --wd 0.0001 --alpha 0.5 --lr_ab 0.0002 > test_on_comb.out 2>test_on_comb.err
python training.py --filter_type high --net StdJacobiSGNNS --jacob_a -0.95 --jacob_b 0.7 --lr 0.05 --wd 0.001 --alpha 1.25 --lr_ab 0.0002 > test_on_high.out 2>test_on_high.err
python training.py --filter_type rejection --net StdJacobiSGNNS --jacob_a -0.80 --jacob_b 0.65 --lr 0.05 --wd 0.0 --alpha 2.5 --lr_ab 0.0002 > test_on_rejection.out 2>test_on_rejection.err
