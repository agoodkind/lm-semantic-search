# Run restart acceptance in an isolated Linux virtual machine

Run every destructive recovery test inside a disposable Linux virtual machine. The harness never connects to production services.

The generic sandbox restores data, starts Docker services, supplies deterministic embeddings, injects faults, records evidence, and cleans tagged resources. LMS-specific code drives Clyde, collections, jobs, searches, and cases A through H.

## Prepare the virtual machine

Use a Linux virtual machine with Docker Engine, Docker Compose, and enough local disk. Any host and virtual machine product may provide the guest.

The validated macOS environment used [Tart](https://tart.run/quick-start/). The same procedure works with another virtual machine system.

Give the guest two CPUs. Store writable acceptance data on the guest disk. Expose only the verified raw backup as a read-only share.

The harness requires free space greater than 125 percent of the archive size. It checks this before restoration and before every case. Stop if the guest lacks space. Do not remove unrelated files.

For Tart, start the guest with runtime values:

```sh
export TART_HOME="${TART_HOME:?set the Tart storage directory}"
export TART_NO_AUTO_PRUNE=1
export VM_NAME="${VM_NAME:?set the Linux virtual machine name}"
export BACKUP_ROOT="${BACKUP_ROOT:?set the verified raw backup directory}"

tart set "$VM_NAME" --cpu 2
tart run \
    --no-graphics \
    --net-softnet \
    --dir "$BACKUP_ROOT:ro,tag=lms-backup" \
    "$VM_NAME"
```

Keep `TART_NO_AUTO_PRUNE=1`. Acceptance work must not remove cached images.

## Prepare copy-on-write storage

The writable run directory must support reflinks. Reflinks create case trees without copying every restored byte.

Use XFS with reflinks enabled or another Linux file system that supports `FICLONE`. A default ext4 root does not satisfy this requirement.

Create a dedicated XFS image when needed. This formats only the new file named by `ACCEPTANCE_DISK_IMAGE`:

```sh
export ACCEPTANCE_DISK_IMAGE="${ACCEPTANCE_DISK_IMAGE:?set the guest XFS image path}"
export ACCEPTANCE_DISK_SIZE="${ACCEPTANCE_DISK_SIZE:?set the XFS image size}"
export LMS_RESTART_ACCEPTANCE_RUN_PARENT="${LMS_RESTART_ACCEPTANCE_RUN_PARENT:?set the guest run directory}"

sudo apt-get install xfsprogs
truncate -s "$ACCEPTANCE_DISK_SIZE" "$ACCEPTANCE_DISK_IMAGE"
sudo mkfs.xfs -f -m reflink=1 "$ACCEPTANCE_DISK_IMAGE"
sudo mkdir -p "$LMS_RESTART_ACCEPTANCE_RUN_PARENT"
sudo mount -o loop "$ACCEPTANCE_DISK_IMAGE" "$LMS_RESTART_ACCEPTANCE_RUN_PARENT"
sudo chown "$USER" "$LMS_RESTART_ACCEPTANCE_RUN_PARENT"
df -h "$LMS_RESTART_ACCEPTANCE_RUN_PARENT"
```

Set the image size above the harness reserve and expected case growth.

## Mount only the backup

Mount the backup read-only inside the guest. For Tart:

```sh
export BACKUP_MOUNT="${BACKUP_MOUNT:?set the guest backup mount directory}"
sudo mkdir -p "$BACKUP_MOUNT"
sudo mount -t virtiofs -o ro lms-backup "$BACKUP_MOUNT"
findmnt "$BACKUP_MOUNT"
```

Stop unless the output includes `ro`. Do not mount a host source tree, Docker socket, service socket, or writable data directory.

## Install Docker and the tested binaries

Install Docker Engine and the Compose plugin with the [Docker Engine Ubuntu instructions](https://docs.docker.com/engine/install/ubuntu/).

Verify the guest engine:

```sh
docker info
docker compose version
docker run --rm hello-world
```

Pull the etcd, MinIO, and Milvus image tags recorded by the backup. The harness checks each local image identifier.

Build and install LMS and Clyde from the exact commits under test:

```sh
make install
lm-semantic-search version
lm-semantic-search-daemon version
clyde --version
```

Stop if a binary reports another commit or a dirty build.

## Run the suite

Set only guest paths:

```sh
export PATH="$HOME/.local/bin:$PATH"
export LMS_RESTART_ACCEPTANCE_BACKUP="${BACKUP_MOUNT:?set the guest backup mount directory}"
export LMS_RESTART_ACCEPTANCE_RUN_PARENT="${LMS_RESTART_ACCEPTANCE_RUN_PARENT:?set the guest run directory}"
```

Run deterministic tests first:

```sh
make restart-acceptance-unit
```

Then provide the exact clone opt-in:

```sh
export LMS_RESTART_ACCEPTANCE_CONFIRM=isolated-clone
make restart-acceptance
```

The harness supplies a local OpenAI-compatible embedding provider. It routes embeddings and cloned Milvus traffic through controllable loopback proxies.

The full run must pass cases A through H without a skip. It then runs clone census, scalar-debt, vector-sample, cold code-search, and cold conversation-search checks.

## Verify cleanup and evidence

The run succeeds only after it removes every tagged container, volume, process, socket, and writable case directory.

Verify Docker after exit:

```sh
docker ps -a --filter name=lms-restart
docker volume ls --filter name=lms-restart
```

Both commands must return no acceptance resource.

Keep the result and event artifacts under the completed run directory. Attach the exact command output and final artifact to the acceptance ticket.

Stop the virtual machine after evidence capture. For Tart:

```sh
tart stop --timeout 60 "$VM_NAME"
```

Keep the virtual machine image and raw backup.

If any case needs manual repair, record the failure. Repeat the complete suite from a clean run.
