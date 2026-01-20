# Ada cluster — User Guide

---

## Overview

Ada cluster consists of ninety two Boston `SYS-7048GR-TR` nodes equipped with dual Intel Xeon E5-2640 v4 processors, providing **40 virtual cores per node**, **128 GB of 2400MT/s DDR4 ECC RAM** and **four Nvidia GeForce GTX 1080 Ti GPUs**, providing **14336 CUDA cores** and **44 GB of GDDR5X VRAM** or **four Nvidia GeForce RTX 2080 Ti GPUs** providing **17408 cores** and **44 GB of GDDR6 VRAM**. The nodes are connected to each other via a Gigabit Ethernet network. All compute nodes have a **1.8 TB local scratch** and a **960 GB local SSD scratch**. The compute nodes are running **Ubuntu 18.04 LTS**. [SLURM](https://slurm.schedmd.com/) software is used as a job scheduler and resource manager. The aggregate theoretical peak performance of Ada is **70.66 TFLOPS (CPU)** + **4588 TFLOPS (FP32 GPU)**.

---

## Applying for an account

An Ada account is available to IIIT faculty, research staff, and research students. To apply for a new account, please send an email to <hpc.admin@iiit.ac.in> with the following information:

- Name
- Roll Number / Employee ID
- Research Center
- Faculty Advisor
- Preferred Login ID

For a CVIT-associated account, fill out [this form](https://forms.gle/L4XdmyHvYQidQged9). The form requires that you sign in using your Gmail account. If you do not get a response about your account creation within **72 hrs** of filling this form, please mail to <cvit-sudo@googlegroups.com> with the same details. If you are a CVIT student, please watch this [tutorial video](https://www.youtube.com/watch?v=U3_pPJgs2Fg).

> **Note:** Strict actions would be taken if requested for a CVIT account without being affiliated to any of the CVIT faculty.

---

## Accessing Ada

### Logging in

You can log in to Ada by SSH from IIIT LAN. For accessing Ada from off-campus, [IIIT VPN](https://vpn.iiit.ac.in/) must be used.

```bash
$ ssh -X user_name@ada.iiit.ac.in
```

### Changing initial password

When you are prompted to enter the password, enter the initial password. In **(current) UNIX password:**, enter the initial password. In **New password:** and **Retype new password:**, enter a new password of your choice.

![Password screenshot](Password2.png)

### Moving files to and from Ada

To move a directory from local machine to Ada:

```bash
$ scp -r local_directory user_name@ada.iiit.ac.in:
# or
$ rsync -avz local_directory user_name@ada.iiit.ac.in:
```

To move a directory from Ada to local machine:

```bash
$ scp -r user_name@ada.iiit.ac.in:remote_directory .
# or
$ rsync -avz user_name@ada.iiit.ac.in:remote_directory .
```

Both `scp` and `rsync` transfer files locally or over network. If the transfer is interrupted, `rsync` has the ability to continue from where it left off when invoked again.

---

## Partitions, Account, and QoS

A partition can be considered as a collection of nodes. There are two primary partitions in the cluster:

- **short** — has two nodes and is for compiling/debugging codes.
- **long** — for serial/parallel jobs that need to run for longer than 6 hours.

Node `01-40` contains **4 GeForce GTX 1080 Ti GPUs** each while nodes `43-92` contain **4 GeForce RTX 2080 Ti GPUs** each.

### Partitions summary

| Partition | Nodes           | DefMemPerCPU | MaxMemPerCPU | Gres   | Maxtime   | Priority |
|-----------|------------------|--------------:|-------------:|:-------|:----------|:---------|
| short     | gnodes[01-02]    | 1024 MB      | 3000 MB      | gpu:4 | 6:00:00   | 100      |
| long      | gnodes[03-92]    | 1024 MB      | 3000 MB      | gpu:4 | Infinite  | 100      |
| ihub      | gnode[92-112]    | 1024        | 5000 MB      | gpu:4 | Infinite  | 100      |
| kcis      | gnode[119-123]   | 1024        | 6000 MB      | gpu:4 | Infinite  | 100      |
| edcs      | gnode[124-127]   | 1024        | 6000 MB      | gpu:4 | Infinite  | 100      |


A SLURM account is like a bank account, and all users belong to at least one account. The allocated resources to a job are charged to the job's specified account. All users have access to the `research` account. The accounts `cvit`, `nlp`, `ccnsb`, `mll`, `hai`, `irel`, `dma`, `adsac`, `rrc`, `plafnet`, and `biosona` are accessible only to users in projects/centres that have contributed hardware to the cluster.

### Accounts summary

| Account  | Access        | GrpCPUs | GrpTRES=gres/gpu | GrpJobs | GrpSubmitJobs | Allowed QoS |
|----------|---------------|--------:|-----------------:|--------:|--------------:|:------------|
| research | ALL           | 1640    | 164              | 820     | 1640          | medium      |
| cvit     | CVIT          | 1080    | 108              | 540     | 1080          | normal      |
| mll      | MLL           | 40      | 4                | 20      | 40            | normal      |
| nlp      | NLP           | 120     | 12               | 60      | 120           | normal      |
| ccnsb    | CCNSB         | 80      | 8                | 40      | 80            | normal      |
| cesp     | CESP          | 40      | 4                | 20      | 40            | normal      |
| hai      | HAI           | 40      | 4                | 20      | 40            | normal      |
| irel     | IREL          | 120     | 12               | 60      | 120           | normal      |
| dma      | DMA           | 40      | 4                | 20      | 40            | normal      |
| adsac    | ADSAC         | 40      | 4                | 20      | 40            | normal      |
| rrc      | RRC           | 40      | 4                | 20      | 40            | normal      |
| plafnet  | PLAfNET       | 40      | 4                | 20      | 40            | normal      |
| biosona  | BioSonA       | 240     | 24               | 120     | 240           | normal      |


It is recommended to specify a Quality of Service (QoS) for each job submitted to SLURM. The default QoS for `research` account is **medium** and it is **normal** for `cvit`, `mll`, `nlp`, `cesp`, `ccnsb`, `hai`, `irel`, `dma`, `adsac`, `rrc`, `plafnet` and `biosona` accounts.

### QoS summary

| QoS    | MaxCPUsPerUser | MaxTRESPerUser    | MaxJobsPerUser | MaxSubmitJobsPerUser | MaxWall    | MaxTresPerJob     | Priority |
|--------|---------------:|------------------:|---------------:|---------------------:|:-----------|:------------------|:---------|
| low    | 10             | gres/gpu=1        | 1              | 4                    | 4-00:00:00 | gres/gpu=1        | 0        |
| medium | 40             | gres/gpu=4        | 4              | 8                    | 4-00:00:00 | gres/gpu=4        | 10       |
| normal | Account limits | Account limits    | Account limits | Account limits       | Infinite   | gres/gpu=4        | 0        |

The following command can be used to list the allowed Accounts and QoSes:

```bash
$ sacctmgr show assoc user=$USER format=Account,QOS,DefaultQOS
```

### CVIT account

To submit jobs to the CVIT account, users should specify SLURM job directive `-A $USER`. Example:

```bash
$ sinteractive -c 2 -g 1 -A $USER
```

The following command will show the associations for an account:

```bash
$ sacctmgr show assoc account=$USER
```

`sinteractive` is non-standard and restrictive in terms of options and documentation. We recommend using `sbatch` and `srun` instead. The following is an example of an `srun` command requesting 1 GPU, 10 CPUs with 2GB memory per CPU (20GB total) with the `cvit` account in the `long` partition. Please maintain a 1:10 ratio for the number of gpu : number of cpu. The CVIT admins may kill jobs in case this is not followed without warning.

```bash
$ srun --pty --partition=long -A $USER --gres=gpu:1 --mem-per-cpu=2G -c 10 bash -l
```

If you want a specific node, use the `--nodelist gnodeXX` parameter. Note that if the requested node is unavailable (other users running jobs, node maintenance, etc.) then your job will go into the `PENDING` state.

Sometimes nodes are reserved (e.g., conference deadlines). Use a reservation name to gain access. Example (reservation-name = `cvit-trial`):

```bash
$ srun --reservation cvit-trial --pty --partition=long -A $USER --gres=gpu:2 --mem-per-cpu=2G -c 10 bash -l
```

`srun` interactive jobs (also called bash jobs) have a max time limit of **6 hrs**. `srun` allocates a terminal for interactive debugging and coding. For jobs requiring more than 6 hrs of runtime, use `sbatch`.

Sample batch script (`batch_script1.sh`):

```bash
#!/bin/bash
#SBATCH -A research/username  # (if you want to use the cvit account put username)
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=op_file.txt

module load u18/cuda/10.1
module load u18/cudnn/7-cuda-10.1

python .....
```

Run the script with:

```bash
$ sbatch batch_script1.sh
```

The output from the python file (anything printed to `stdout`) will be stored in `op_file.txt`. Please use `flush=True` in `print` statements to flush the buffer regularly. To continually check updates to the file, use:

```bash
$ tail -f op_file.txt
```

CVIT admins control `GPUMinutes` and the maximum number of GPUs allocated to a user. By default on account creation, **1 GPU** and **600 GPUMinutes** is allowed to a user. Please mail to <cvit-sudo@googlegroups.com> to change these limits.

### sub account

A new SLURM account `sub` has been created to utilize idle nodes/cores/GPUs. Details:

```
SLURM account: sub  (#SBATCH -A sub)
QOS:     sub        (#SBATCH --qos=sub)
Wall time: 6 hours  (#SBATCH --time=6:00:00)
MaxCPUs: 40
Max GPUs: 4
```

Jobs submitted using this account will have low priority and will run only when there are no high priority (QoS: normal, medium) pending jobs.

---

## File Systems

The cluster provides four types of file storage to users: `/home`, `/share1`, `/scratch` and `/ssd_scratch`. For CVIT users, there is `/share3` for long-term storage.

- `/home` located at `/home/$USER` is an NFS storage and has a disk quota of **25 GB**. This space is for source code and building executables. `/home` is backed up every day.
- `/share1` is RAID6 storage available on the master node and has a quota of **100 GB** (CVIT users have a group quota of **6 TB**). Public datasets are stored under `/share1/dataset`.
- `/share3` is long-term storage for CVIT users. Default per-user space is **50 GB** (can be increased to 200 GB on request).
- `/scratch` is for storing temporary files created during job run time (local disk attached to each compute node).
- `/ssd_scratch` is for storing temporary files that require high-speed disk I/O (local SSD on each compute node).

Files older than **10 days** are purged from `/scratch` and `/ssd_scratch`.

### File systems summary

| Space      | Purpose                                    | Visibility                  | Backup | Quota                  | Total Size | File Deletion Policy |
|------------|--------------------------------------------|----------------------------:|:------:|:-----------------------|----------:|:----------------------|
| /home      | Software installation space, codes, small files | Master and compute nodes | Yes    | 25 GB                  | 9.8 TB    | None                 |
| /share1    | Long-term storage                           | Master node only          | No     | 100 GB / 6 TB (CVIT)   | 20 TB     | None                 |
| /share2    | Long-term storage                           | Master node only          | No     |                       | 13 TB     | None                 |
| /share3    | Long-term storage (CVIT only)               | Master node only          | No     | 50/100/150/200 GB      | 22 TB     | None                 |
| /scratch   | Temporary storage for large files           | Local disk on compute node | No     | None                   | 2.0 TB    | 7 days (based on ctime) |
| /ssd_scratch | Temporary storage for fast I/O jobs       | Local disk on compute node | No     | None                   | 960 GB    | 7 days (based on ctime) |

> Files are deleted from `/scratch` and `/ssd_scratch` based on creation time (ctime).

---

## Policies

### Account Policies

1. Access to HPC account is open to faculty members, post-doctoral researchers and research students (MS and PhD). Under certain circumstances, a non-research student may be granted access with endorsement of their faculty advisor.
2. To apply for an account, send an email to <hpc@iiit.ac.in> with: Name, Roll number / Employee ID, Research Center, Faculty Advisor and Preferred login name.
3. Users are required to change their password every six months. Alerts start one month before expiry.
4. Accounts will be removed/deleted when a student has been issued no-dues certificate by IT office and when an employee leaves the institute. Accounts not accessed for six months will be locked.
5. Sharing accounts is strictly forbidden and will result in the account being locked for 3 days to a week.
6. The account of a student will be locked upon request of their faculty advisor; an employee's account will be locked upon request of a competent authority.

### Software Policies

1. Custom Python and R environments must be installed in users' home directories.
2. Most commonly used software is installed system-wide and available as modules. For specific software installation requests, mail <hpc@iiit.ac.in>.
3. Installation of any unlicensed proprietary software on HPC nodes is strictly prohibited.

### Security Policies

1. Sharing of accounts is strictly prohibited.
2. Users are advised to set up passwordless SSH access to their accounts for convenience.
3. Users should not leave their terminal unattended while logged in.
4. Any suspicious activities or security problems should be reported to <hpc@iiit.ac.in> immediately.

### Backup Policies

1. Users' home directories are backed up once every 24 hours; backups are retained for up to three months. Users are strongly advised to maintain additional backups of important data.
2. The backups are accessible as read-only on the login node.

---

## Environment Modules

The environment module allows users to set shell environment variables needed for software.

To view the list of currently loaded modules:

```bash
[parithi@ada ~]$ module list

Currently Loaded Modulefiles:
  1) u18/namd/2.14       2) u18/openmpi/4.1.2
```

To list installed modules:

```bash
[parithi@ada ~]$ module avail
```

Sample (truncated) output shows many `u18/*` modules, for example `u18/cuda/11.6`, `u18/python/3.10.2`, `u18/openmpi/4.1.2`, etc.

To load a module:

```bash
[parithi@ada ~]$ module load u18/openmpi/4.1.2
```

To remove/unload a module:

```bash
[parithi@ada ~]$ module unload u18/openmpi/4.1.2
```

To display the changes a module makes to the shell environment:

```bash
[parithi@ada ~]$ module disp u18/cuda/11.6
-------------------------------------------------------------------
/opt/Modules/3.2.10/modulefiles/u18/cuda/11.6:

module-whatis	 adds CUDA-11.6 to your environment variable
prepend-path	 PATH /usr/local/cuda-11.6/bin
prepend-path	 LD_LIBRARY_PATH /usr/local/cuda-11.6/lib64
-------------------------------------------------------------------
```

### List of available modules (selection)

| Software | Module | Remarks |
|----------|--------|--------:|
| CUDA 9.0 | `cuda/9.0` | |
| CUDA 9.1 | `cuda/9.1` | |
| CUDA 10.0 | `cuda/10.0` | |
| cuDNN 7 | `cudnn/7-cuda-9.0` | CUDA 9.0 |
| cuDNN 7.6.4 | `cudnn/7.6.4-cuda-9.0` | CUDA 9.0 |
| cuDNN 7.1 | `cudnn/7.1-cuda-9.1` | CUDA 9.1 |
| cuDNN 7 | `cudnn/7-cuda-10.0` | CUDA 10.0 |
| cuDNN 7.3 | `cudnn/7.3-cuda-10.0` | CUDA 10.0 |
| cuDNN 7.6 | `cudnn/7.6-cuda-10.0` | CUDA 10.0 |
| OpenBLAS | `openblas/0.3.6` | Haswell |
| OpenCV | `opencv/3.3.0` | Python 2.7 |
| OpenMPI | `openmpi/2.1.1` | |
| OpenMPI | `openmpi/3.1.0` | |
| OpenMPI | `openmpi/4.0.0` | |
| OpenMPI | `openmpi/4.0.1-cuda10` | CUDA aware |
| NAMD 2.13 | `namd/2.13` | |
| PLUMED 2.5.2 | `plumed/2.5.2` | |
| GROMACS 2019 | `gromacs/2019` | |
| GROMACS 2019.3 | `gromacs/2019.3-plumed` | plumed patched |
| AMBER 16 | `amber/16` | Compiled with GCC-4.7 |
| gflags | `gflags/2.2.1`, `gflags/2.2.2` | |
| glog | `glog/0.3.5`, `glog/0.4.0` | |
| MATLAB | `matlab/R2019b` | |
| MPICH | `mpich-3.2` | Compiled with GCC-5 |
| Intel TBB | `tbb/2018u1-release` | Compiled with macro TBB_USE_DEBUG=0 |
| Intel TBB | `tbb/2018u1-debug` | Compiled with macro TBB_USE_DEBUG=1 |
| FFMPEG | `ffmpeg/3.4`, `ffmpeg/4.01` | `ffmpeg/3.4` has cuda 9.0, cuvid, nvenc, nonfree, libnpp |

---

## SLURM Commands

| Description                      | Command (link)                                  | Example                          |
|----------------------------------|------------------------------------------------|----------------------------------|
| Submit a batch job               | [sbatch](https://slurm.schedmd.com/sbatch.html) | `sbatch job_script.sh`           |
| Submit an interactive job        | `sinteractive`                                  | `sinteractive -c 2 -g 1`         |
| Cancel a job                     | [scancel](https://slurm.schedmd.com/scancel.html) | `scancel job_id`               |
| List all current jobs for a user | [squeue](https://slurm.schedmd.com/squeue.html) | `squeue -u username`            |
| Statistics of a completed job    | [sacct](https://slurm.schedmd.com/sacct.html)  | `sacct -j job_id --format=user,jobid,jobname,partition,state,time,start,end,elapsed,alloctres,ncpus,nodelist` |
| Pause a job                      | [scontrol](https://slurm.schedmd.com/scontrol.html) | `scontrol hold job_id`     |
| Resume a job                     | [scontrol](https://slurm.schedmd.com/scontrol.html) | `scontrol resume job_id`   |
| Modify attributes of a job       | [scontrol](https://slurm.schedmd.com/scontrol.html) | `scontrol update jobid=job_id TimeLimit=4-00:00:00` |
| Display a job's characteristics  | [scontrol](https://slurm.schedmd.com/scontrol.html) | `scontrol show job job_id`  |
| Display information about nodes and partitions | [sinfo](https://slurm.schedmd.com/sinfo.html) | `sinfo -a`                 |


---

## SLURM Job Pending Reasons

- **None** — The job is in the queue, but the SLURM controller has not yet determined whether it is pending due to priority, resource availability or another constraint. A pending reason will be assigned shortly.
- **Priority** — The job is waiting for its turn based on priority.
- **ReqNodeNotAvail** — A required node is unavailable (e.g., down, drained, or in maintenance).
- **Resources** — There are not enough free resources (CPUs, memory, GPUs, etc.).
- **QOSMaxJobsLimit** — The job is exceeding the allowed number of jobs per user/account in QoS policy.
- **QOSMaxCpuPerUserLimit** — The job is exceeding the allowed number of CPU cores per user in QoS policy.
- **AssocGrpGRES** — The job is exceeding the group resources (CPU, GPU etc.).
- **JobHeldAdmin** — The job is on hold due to an admin action.
- **JobHeldUser** — The job is on hold due to a user action.
- **PartitionDown** — The assigned partition is unavailable.
- **PartitionNodeLimit** — The job requests more nodes than allowed in the partition.
- **AssocGrpCpuLimit** — The job's account or group has hit its CPU limit.
- **AssocGrpMemLimit** — The job's account or group has hit its memory limit.

---

## Job Submission

### Interactive Jobs

Example: Cores = 10, partition = long, Account = research, GPU = 1

```bash
[parithi@ada ~]$ sinteractive -c 10 -p long -A research -g 1
salloc: Granted job allocation 141
parithi@gnode03:/home/parithi$
```

### Batch Jobs

#### Sample script: NAMD

```bash
#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END

module add namd/2.12

scp ada:/share1/$USER/file.tar

charmrun +p$SLURM_NPROCS namd2 +idlepoll +devices $CUDA_VISIBLE_DEVICES apoa1.namd > output-gpu1.out
```

#### Sample script: Python

```bash
#!/bin/bash
#SBATCH -A $USER
#SBATCH -n 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END

module add cuda/8.0
module add cudnn/7-cuda-8.0

scp ada:/share1/$USER/file.tar /scratch

tar xf /scratch/file.tar
python .....

scp /scratch/output  ada:/share1/$USER
```

The variable `CUDA_VISIBLE_DEVICES` holds the ids of assigned GPUs.

---

## Storage and usage tutorial video

The below video can help you to get important details about the usage of Ada Machine:

https://www.youtube.com/watch?v=U3_pPJgs2Fg

---

## Using JupyterLab / TensorBoard on compute nodes

**Steps:**

1. Set JupyterLab password (required once):

```bash
jupyter lab password
```

2. Start JupyterLab / TensorBoard on a compute node on any port.

3. From your local machine run an SSH tunnel (example):

```bash
ssh -L port1:localhost:port2 -J username@ada.iiit.ac.in username@gnodeXX
```

- `port1`: port on your local machine (laptop/PC)
- `port2`: port on the compute node where JupyterLab/TensorBoard is running

Now access JupyterLab/TensorBoard at `localhost:port1` on your local machine.

**Note:** SSH server is required on the local machine for this method. For Windows, you can use [Bitvise](https://www.bitvise.com/ssh-serve). You can also add reverse SSH tunnel config in `~/.ssh/config`.

---

## Windows Users

### Connecting to Ada

Connect to [IIIT VPN](https://vpn.iiit.ac.in/) first. Then run:

```bash
ssh -X username@ada.iiit.ac.in
```

You will be in the head-node of Ada, inside your `/home2/username/` directory.

### Mounting Ada to Local

Windows users can mount their Ada drive using [SFTP Drive V2](https://www.nsoftware.com/sftp/drive/), which is free for personal use. Configure a new drive with the following:

- **Remote Host:** `ada.iiit.ac.in`
- **Username and Password:** Your Ada credentials
- **Root folder on Server:** `/home2/username`

Start the drive to mount the remote server locally.

---

## Accessing compute nodes via VSCode

1. Install the **RemoteSSH** extension in VS Code on your local machine.
2. Launch an interactive or batch job on Ada and note the assigned compute node.
3. Edit `~/.ssh/config` on your local machine to include entries like:

```
Host ada
  HostName ada.iiit.ac.in
  User your_Ada_user_id

Host gnode
  HostName gnodexxx
  User your_Ada_user_id

ProxyCommand ssh -W %h:%p ada
```

4. In VS Code, use RemoteSSH to connect to `gnode`.

---

*End of document.*

