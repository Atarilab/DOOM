#!/bin/bash
# Script to setup and test network connection to robot
# Usage: ./setup_robot_network.sh [interface_name]
# Example: ./setup_robot_network.sh enx3c4937046061

set -e

ROBOT_IP="192.168.123.161"
CONTROLLER_IP="192.168.123.11"
NETMASK="255.255.255.0"

# Allow interface name to be passed as argument
if [[ -n "$1" ]]; then
    ETH_IFACE="$1"
    echo "=== Robot Network Setup and Diagnostics ==="
    echo "Using specified interface: $ETH_IFACE"
    echo ""
else
    echo "=== Robot Network Setup and Diagnostics ==="
    echo ""

    # Find ethernet interface
    echo "1. Checking network interfaces..."
    IFACES=$(ip link show | grep -E "^[0-9]+:" | grep -v "lo:" | awk -F': ' '{print $2}' | awk '{print $1}')
    echo "Available interfaces:"
    for iface in $IFACES; do
        ip_addr=$(ip -4 addr show $iface 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' || echo "no IP")
        echo "  - $iface: $ip_addr"
    done
    echo ""

    # Ask user to select interface or auto-detect
    echo "2. Auto-detecting ethernet interface..."
    ETH_IFACE=""
    for iface in $IFACES; do
        # Check if it's an ethernet interface (not wifi, not lo, not docker)
        # Include enx* interfaces (USB ethernet adapters)
        if [[ ! "$iface" =~ ^(wlan|wlp|docker|br-|veth) ]] && [[ "$iface" != "lo" ]]; then
            # Check if it's not already configured with a different IP in 192.168.123.x
            current_ip=$(ip -4 addr show $iface 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' || echo "")
            if [[ -n "$current_ip" ]]; then
                if [[ "$current_ip" =~ ^192\.168\.123\. ]]; then
                    ETH_IFACE="$iface"
                    echo "  Found ethernet interface: $iface (already configured with $current_ip)"
                    break
                fi
            else
                # Prefer eth0, enp*, eno*, or enx* interfaces (USB ethernet adapters)
                if [[ "$iface" =~ ^(eth0|enp|eno|enx) ]] || [[ -z "$ETH_IFACE" ]]; then
                    ETH_IFACE="$iface"
                    echo "  Found ethernet interface: $iface"
                fi
            fi
        fi
    done

    if [[ -z "$ETH_IFACE" ]]; then
        echo "ERROR: Could not auto-detect ethernet interface."
        echo "Please specify the interface name manually:"
        read -p "Interface name (e.g., eth0, enp3s0, enx3c4937046061): " ETH_IFACE
    fi
fi

echo "Using interface: $ETH_IFACE"
echo ""

# Check current configuration
echo "3. Checking current interface configuration..."
CURRENT_IP=$(ip -4 addr show $ETH_IFACE 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' || echo "not configured")

if [[ "$CURRENT_IP" == "$CONTROLLER_IP" ]]; then
    echo "  Interface $ETH_IFACE is already configured with $CONTROLLER_IP"
else
    echo "  Current IP: $CURRENT_IP"
    echo "  Configuring interface $ETH_IFACE with static IP $CONTROLLER_IP..."
    
    # Bring interface down
    ip link set $ETH_IFACE down 2>/dev/null || true
    
    # Configure static IP
    ip addr flush dev $ETH_IFACE 2>/dev/null || true
    ip addr add $CONTROLLER_IP/$NETMASK dev $ETH_IFACE 2>/dev/null || {
        echo "ERROR: Failed to configure IP. You may need to run this script with sudo or as root."
        exit 1
    }
    
    # Bring interface up
    ip link set $ETH_IFACE up
    
    echo "  Successfully configured $ETH_IFACE with IP $CONTROLLER_IP"
fi
echo ""

# Verify interface is up
echo "4. Verifying interface status..."
if ip link show $ETH_IFACE | grep -q "state UP"; then
    echo "  Interface $ETH_IFACE is UP"
else
    echo "  WARNING: Interface $ETH_IFACE appears to be DOWN"
    echo "  Attempting to bring it up..."
    ip link set $ETH_IFACE up
    sleep 1
fi
echo ""

# Test connectivity
echo "5. Testing connectivity to robot ($ROBOT_IP)..."
echo "  Pinging robot..."
if ping -c 3 -W 2 $ROBOT_IP > /dev/null 2>&1; then
    echo "  ✓ SUCCESS: Can ping robot at $ROBOT_IP"
    ping -c 3 $ROBOT_IP | tail -n 2
else
    echo "  ✗ FAILED: Cannot ping robot at $ROBOT_IP"
    echo ""
    echo "Troubleshooting steps:"
    echo "  1. Check that robot is powered on and ethernet cable is connected"
    echo "  2. Verify robot IP is actually $ROBOT_IP (check robot configuration)"
    echo "  3. Check routing: ip route show"
    echo "  4. Check firewall: iptables -L -n | grep $ROBOT_IP"
    echo "  5. Try: ip route add $ROBOT_IP/32 dev $ETH_IFACE"
    echo "  6. Check interface link status: ethtool $ETH_IFACE"
    exit 1
fi
echo ""

# Show routing
echo "6. Checking routes to robot network..."
ip route show | grep -E "(192\.168\.123|$ROBOT_IP)" || echo "  No specific route found (using default route)"
echo ""

echo "=== Network setup complete! ==="
echo "Interface: $ETH_IFACE"
echo "Controller IP: $CONTROLLER_IP"
echo "Robot IP: $ROBOT_IP"
echo ""
echo "You should now be able to communicate with the robot from the container."

