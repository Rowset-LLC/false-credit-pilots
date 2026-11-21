#!/bin/bash
# Detached runner: VM setup + pilot. Started with PowerShell Start-Process so the PC memory reaper can't kill it.
cd "$(dirname "$0")"
IP=$1; K=${SSH_KEY:-$HOME/.ssh/id_ed25519}; O="-i $K -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
ssh-keygen -R $IP >/dev/null 2>&1
for i in $(seq 1 90); do ssh $O ubuntu@$IP true 2>/dev/null && break; sleep 10; done
echo "ssh up $(date -u +%H:%M:%S)"
ssh $O ubuntu@$IP 'mkdir -p ~/p ~/data'
scp $O vm_train.py ubuntu@$IP:~/p/vm_train.py
ssh $O ubuntu@$IP 'while sudo fuser /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock /var/lib/dpkg/lock >/dev/null 2>&1; do sleep 10; done; echo "apt free $(date -u +%H:%M:%S)"; sudo apt-get -qq update >/dev/null 2>&1; for i in 1 2 3 4 5; do sudo apt-get -qq install -y python3-venv >/dev/null 2>&1 && break; sleep 30; done; python3 -m venv ~/venv && ~/venv/bin/pip -q install torch torchvision >/dev/null 2>&1; ~/venv/bin/python -c "import torch,torchvision;print(torch.__version__,torch.cuda.is_available()); torchvision.datasets.CIFAR10(\"/home/ubuntu/data\",train=True,download=True); torchvision.datasets.CIFAR10(\"/home/ubuntu/data\",train=False,download=True); print(\"data ok\")" 2>&1 | grep -v "%"'
ssh $O ubuntu@$IP '~/venv/bin/python -c "import torch; assert torch.cuda.is_available()"' || { echo "SETUP_FAILED $(date -u +%H:%M:%S)"; echo "PILOT_DONE $(date -u +%H:%M:%S)"; exit 1; }
echo "setup done $(date -u +%H:%M:%S)"
python pilot3.py --ip $IP --rounds 10 --repeats 2 --arms A,B --cap 3.0
echo "PILOT_DONE $(date -u +%H:%M:%S)"
