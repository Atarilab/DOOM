#!/bin/bash
# Quick network diagnostic script for robot connectivity

ROBOT_IP="192.168.123.161"
CONTROLLER_IP="192.168.123.11"

echo "=== Network Diagnostic ==="
echo ""

echo "1. Checking if running in container with host network..."
if [ -f /.dockerenv ]; then
    echo "  ✓ Running in Docker container"
    if docker inspect $(hostname) 2>/dev/null | grep -q '"NetworkMode": "host"'; then
        echo "  ✓ Container is using host network mode"
    else
        echo "  ✗ WARNING: Container may not be using host network mode"
        echo "     Check docker-compose.yml for: network_mode: host"
    fi
else
    echo "  Running on host (not in container)"
fi
echo ""

echo "2. Network interfaces:"
ip -4 addr show | grep -E "^[0-9]+:|inet " | while read line; do
    if [[ $line =~ ^[0-9]+: ]]; then
        echo "  $line"
    elif [[ $line =~ inet ]]; then
        echo "    $line"
    fi
done
echo ""

echo "3. Looking for ethernet interface with robot network (192.168.123.x)..."
FOUND=false
for iface in $(ip link show | grep -E "^[0-9]+:" | grep -v "lo:" | awk -F': ' '{print $2}' | awk '{print $1}'); do
    ip_addr=$(ip -4 addr show $iface 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' || echo "")
    if [[ -n "$ip_addr" ]] && [[ "$ip_addr" =~ ^192\.168\.123\. ]]; then
        echo "  ✓ Found: $iface with IP $ip_addr"
        FOUND=true
        
        # Check link status
        if ip link show $iface | grep -q "state UP"; then
            echo "    Status: UP"
        else
            echo "    Status: DOWN (needs to be brought up)"
        fi
        
        # Check link detection
        if command -v ethtool >/dev/null 2>&1; then
            link_detected=$(ethtool $iface 2>/dev/null | grep "Link detected" | awk '{print $3}')
            if [[ "$link_detected" == "yes" ]]; then
                echo "    Link detected: yes"
            else
                echo "    Link detected: no (check cable connection)"
            fi
        fi
    fi
done

if [[ "$FOUND" == "false" ]]; then
    echo "  ✗ No interface found with IP in 192.168.123.x subnet"
    echo "     You may need to configure an interface with IP $CONTROLLER_IP"
fi
echo ""

echo "4. Routing table:"
ip route show | grep -E "(192\.168\.123|default)" || echo "  No relevant routes found"
echo ""

echo "5. Testing connectivity to robot ($ROBOT_IP)..."
if ping -c 1 -W 2 $ROBOT_IP >/dev/null 2>&1; then
    echo "  ✓ SUCCESS: Can ping robot at $ROBOT_IP"
    echo "  Response times:"
    ping -c 3 $ROBOT_IP | tail -n 2
else
    echo "  ✗ FAILED: Cannot ping robot at $ROBOT_IP"
    echo ""
    echo "  Troubleshooting steps:"
    echo "  1. Ensure ethernet cable is connected to robot"
    echo "  2. Configure ethernet interface with IP $CONTROLLER_IP:"
    echo "     sudo ip addr add $CONTROLLER_IP/24 dev <interface>"
    echo "     sudo ip link set <interface> up"
    echo "  3. Check robot is powered on and configured with IP $ROBOT_IP"
    echo "  4. Try from host (outside container) to verify robot connectivity"
    echo "  5. Check firewall rules: sudo iptables -L -n"
fi
echo ""

echo "6. Checking for potential issues..."
# Check if NetworkManager might be interfering
if command -v nmcli >/dev/null 2>&1; then
    for iface in $(ip link show | grep -E "^[0-9]+:" | grep -v "lo:" | awk -F': ' '{print $2}' | awk '{print $1}'); do
        if [[ "$iface" =~ ^(eth|enp|eno) ]]; then
            nm_status=$(nmcli device status 2>/dev/null | grep "$iface" | awk '{print $3}' || echo "")
            if [[ "$nm_status" == "managed" ]]; then
                echo "  ⚠ NetworkManager is managing $iface - may interfere with manual config"
                echo "     Consider: sudo nmcli device set $iface managed no"
            fi
        fi
    done
fi

# Check firewall
if command -v ufw >/dev/null 2>&1; then
    ufw_status=$(ufw status 2>/dev/null | head -n 1 | grep -o "active" || echo "")
    if [[ "$ufw_status" == "active" ]]; then
        echo "  ⚠ UFW firewall is active - may block ping"
    fi
fi

echo ""
echo "=== Diagnostic complete ==="

