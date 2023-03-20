python RealWorld.py --test --repeat 10 --split dense --dataset chameleon --learnable_bases --wd4 0.0005 --lr4 0.001 --path "results" --name "exp_on_chameleon.out" >test_on_chameleon.out 2>test_on_chameleon.err

python RealWorld.py --test --repeat 10 --split dense --dataset citeseer --learnable_bases --wd4 0.0001 --lr4 0.001 --path "results" --name "exp_on_citeseer.out" >test_on_citeseer.out 2>test_on_citeseer.err

python RealWorld.py --test --repeat 10 --split dense --dataset computers --learnable_bases --wd4 0.001 --lr4 0.01 --path "results" --name "exp_on_computers.out" >test_on_computers.out 2>test_on_computers.err

python RealWorld.py --test --repeat 10 --split dense --dataset cora --learnable_bases --wd4 0.001 --lr4 0.001 --path "results" --name "exp_on_cora.out" >test_on_cora.out 2>test_on_cora.err

python RealWorld.py --test --repeat 10 --split dense --dataset cornell --learnable_bases --wd4 0.0005 --lr4 0.01 --path "results" --name "exp_on_cornell.out" >test_on_cornell.out 2>test_on_cornell.err

python RealWorld.py --test --repeat 10 --split dense --dataset film --learnable_bases --wd4 0.001 --lr4 0.01 --path "results" --name "exp_on_film.out" >test_on_film.out 2>test_on_film.err

python RealWorld.py --test --repeat 10 --split dense --dataset photo --learnable_bases --wd4 0.0005 --lr4 0.001 --path "results" --name "exp_on_photo.out" >test_on_photo.out 2>test_on_photo.err

python RealWorld.py --test --repeat 10 --split dense --dataset pubmed --learnable_bases --wd4 0.0005 --lr4 0.001 --path "results" --name "exp_on_pubmed.out" >test_on_pubmed.out 2>test_on_pubmed.err

python RealWorld.py --test --repeat 10 --split dense --dataset squirrel --learnable_bases --wd4 0.0005 --lr4 0.001 --path "results" --name "exp_on_squirrel.out" >test_on_squirrel.out 2>test_on_squirrel.err

python RealWorld.py --test --repeat 10 --split dense --dataset texas --learnable_bases --wd4 0.0001 --lr4 0.001 --path "results" --name "exp_on_texas.out" >test_on_texas.out 2>test_on_texas.err
