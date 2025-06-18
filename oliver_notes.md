Add this to `.vscode/launch.json` for debug ROS node:
```
        {
            "name": "Velocity Sim Go2 debug (remote attach)",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "${workspaceFolder}"
                }
            ],
            "justMyCode": false
        },
```

and start like this:

`ros2 run master_manager master_node --task rl-velocity-sim-go2 --log test --enable-ui --debug`

Run `pip install debugpy` in container