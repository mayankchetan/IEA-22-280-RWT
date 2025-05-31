#!/bin/bash
# kestrel_move_script.sh
# Simple interactive script to transfer files to/from Kestrel
# Load configuration from secret file

# Source the configuration file
source _secret.conf

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Kestrel Transfer Utility ===${NC}"

# Function to send files to Kestrel
send_to_kestrel() {
    echo -e "${BLUE}== Send Files to Kestrel ==${NC}"


    # Confirm transfer
    echo "Will transfer: $LOCAL_DIR → $KESTREL_USER@$KESTREL_ADDRESS:$REMOTE_DIR"
    read -p "Proceed? (y/n): " confirm
    
    if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
        echo -e "${YELLOW}Transferring files to Kestrel...${NC}"
        
        # Create the destination directory if it doesn't exist
        ssh $KESTREL_USER@$KESTREL_ADDRESS "mkdir -p $REMOTE_DIR"
        
        # Transfer the files
        rsync -avz -rh --progress --update "$LOCAL_DIR" $KESTREL_USER@$KESTREL_ADDRESS:$REMOTE_DIR/
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Transfer completed successfully${NC}"
        else
            echo "Transfer failed"
        fi
    else
        echo "Transfer canceled"
    fi
}

# Function to retrieve files from Kestrel
get_from_kestrel() {
    echo -e "${BLUE}== Get Files from Kestrel ==${NC}"
    
    # List files on Kestrel project directory
    echo -e "${YELLOW}Listing available files on Kestrel:${NC}"
    
    ssh $KESTREL_USER@$KESTREL_ADDRESS "ls -la $REMOTE_DIR"
    
    # Create the destination directory if it doesn't exist
    mkdir -p "$LOCAL_DIR"
    
    # Confirm transfer
    echo "Will transfer: $KESTREL_USER@$KESTREL_ADDRESS:$REMOTE_DIR/$LOCAL_DIR/ → $LOCAL_DIR"
    read -p "Proceed? (y/n): " confirm
    
    if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
        echo -e "${YELLOW}Retrieving files from Kestrel...${NC}"
        rsync -avz -rh --progress --update $KESTREL_USER@$KESTREL_ADDRESS:$REMOTE_DIR/$LOCAL_DIR/ "$LOCAL_DIR"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Transfer completed successfully${NC}"
        else
            echo "Transfer failed"
        fi
    else
        echo "Transfer canceled"
    fi
}

# Main menu
while true; do
    echo -e "\n${BLUE}=== Kestrel Transfer Menu ===${NC}"
    echo "1. Send files to Kestrel"
    echo "2. Get files from Kestrel"
    echo "3. Exit"
    read -p "Choose an option (1-3): " option
    
    case $option in
        1) send_to_kestrel ;;
        2) get_from_kestrel ;;
        3) echo "Exiting."; exit 0 ;;
        *) echo "Invalid option. Please try again." ;;
    esac
done