#!/bin/bash

# HarpoonAI Desktop Shortcuts Creator for Fedora Plasma/KDE
# Creates desktop shortcuts for starting, stopping, and managing HarpoonAI

PROJECT_DIR="$HOME/offline_ai_chat"
DESKTOP_DIR="$HOME/Desktop"
APPLICATIONS_DIR="$HOME/.local/share/applications"
ICONS_DIR="$HOME/.local/share/icons"

echo "🎨 Creating HarpoonAI Desktop Shortcuts..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if HarpoonAI is installed
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ Error: HarpoonAI not found at $PROJECT_DIR"
    echo "Please run the installation script first."
    exit 1
fi

# Check if required scripts exist
if [ ! -f "$PROJECT_DIR/start_harpoonai.sh" ]; then
    echo "❌ Error: start_harpoonai.sh not found"
    echo "Please complete the HarpoonAI installation first."
    exit 1
fi

echo "🎨 Creating HarpoonAI Desktop Shortcuts..."

# Create necessary directories if they don't exist
echo "📁 Creating required directories..."
mkdir -p "$ICONS_DIR"
mkdir -p "$APPLICATIONS_DIR"
mkdir -p "$DESKTOP_DIR"

# Check if directories were created successfully
if [ ! -d "$APPLICATIONS_DIR" ]; then
    echo "❌ Error: Could not create $APPLICATIONS_DIR"
    echo "Trying alternative location..."
    APPLICATIONS_DIR="$HOME/.local/share/applications"
    mkdir -p "$APPLICATIONS_DIR"
fi

if [ ! -d "$ICONS_DIR" ]; then
    echo "❌ Error: Could not create $ICONS_DIR"
    echo "Trying alternative location..."
    ICONS_DIR="$HOME/.local/share/icons"
    mkdir -p "$ICONS_DIR"
fi

echo "✅ Directories created:"
echo "   • Applications: $APPLICATIONS_DIR"
echo "   • Icons: $ICONS_DIR"
echo "   • Desktop: $DESKTOP_DIR"

# Create a simple SVG icon for HarpoonAI (Start)
cat > "$ICONS_DIR/harpoonai-start.svg" << 'ICONEOF'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0ea5e9;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#22c55e;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#a855f7;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="512" height="512" rx="80" fill="#000"/>
  <rect width="492" height="492" x="10" y="10" rx="70" fill="url(#grad1)" opacity="0.1"/>
  <path d="M 256 128 L 384 256 L 256 384 L 256 320 L 128 320 L 128 192 L 256 192 Z" fill="url(#grad1)"/>
  <circle cx="256" cy="256" r="200" fill="none" stroke="url(#grad1)" stroke-width="8" opacity="0.5"/>
  <text x="256" y="460" font-family="Arial, sans-serif" font-size="36" font-weight="bold" fill="url(#grad1)" text-anchor="middle">HarpoonAI</text>
</svg>
ICONEOF

# Create a simple SVG icon for HarpoonAI (Stop)
cat > "$ICONS_DIR/harpoonai-stop.svg" << 'ICONEOF'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="grad2" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#dc2626;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#f97316;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="512" height="512" rx="80" fill="#000"/>
  <rect width="492" height="492" x="10" y="10" rx="70" fill="url(#grad2)" opacity="0.1"/>
  <rect x="176" y="176" width="160" height="160" rx="20" fill="url(#grad2)"/>
  <circle cx="256" cy="256" r="200" fill="none" stroke="url(#grad2)" stroke-width="8" opacity="0.5"/>
  <text x="256" y="460" font-family="Arial, sans-serif" font-size="36" font-weight="bold" fill="url(#grad2)" text-anchor="middle">HarpoonAI</text>
</svg>
ICONEOF

# Create a simple SVG icon for HarpoonAI (Status)
cat > "$ICONS_DIR/harpoonai-status.svg" << 'ICONEOF'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="grad3" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#3b82f6;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#8b5cf6;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="512" height="512" rx="80" fill="#000"/>
  <rect width="492" height="492" x="10" y="10" rx="70" fill="url(#grad3)" opacity="0.1"/>
  <circle cx="256" cy="256" r="80" fill="none" stroke="url(#grad3)" stroke-width="16"/>
  <circle cx="256" cy="120" r="12" fill="url(#grad3)"/>
  <circle cx="352" cy="176" r="12" fill="url(#grad3)"/>
  <circle cx="392" cy="256" r="12" fill="url(#grad3)"/>
  <circle cx="352" cy="336" r="12" fill="url(#grad3)"/>
  <circle cx="256" cy="392" r="12" fill="url(#grad3)"/>
  <circle cx="160" cy="336" r="12" fill="url(#grad3)"/>
  <circle cx="120" cy="256" r="12" fill="url(#grad3)"/>
  <circle cx="160" cy="176" r="12" fill="url(#grad3)"/>
  <text x="256" y="460" font-family="Arial, sans-serif" font-size="36" font-weight="bold" fill="url(#grad3)" text-anchor="middle">HarpoonAI</text>
</svg>
ICONEOF

# Create a simple SVG icon for HarpoonAI (Open Browser)
cat > "$ICONS_DIR/harpoonai-browser.svg" << 'ICONEOF'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="grad4" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#10b981;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#06b6d4;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="512" height="512" rx="80" fill="#000"/>
  <rect width="492" height="492" x="10" y="10" rx="70" fill="url(#grad4)" opacity="0.1"/>
  <circle cx="256" cy="256" r="120" fill="none" stroke="url(#grad4)" stroke-width="16"/>
  <path d="M 256 136 C 256 136, 256 376, 256 376" stroke="url(#grad4)" stroke-width="8"/>
  <path d="M 136 256 C 136 256, 376 256, 376 256" stroke="url(#grad4)" stroke-width="8"/>
  <path d="M 180 180 Q 256 160, 332 180" fill="none" stroke="url(#grad4)" stroke-width="8"/>
  <path d="M 180 332 Q 256 352, 332 332" fill="none" stroke="url(#grad4)" stroke-width="8"/>
  <text x="256" y="460" font-family="Arial, sans-serif" font-size="36" font-weight="bold" fill="url(#grad4)" text-anchor="middle">HarpoonAI</text>
</svg>
ICONEOF

# Create Start Desktop Entry
echo "📝 Creating Start shortcut..."
cat > "$APPLICATIONS_DIR/harpoonai-start.desktop" << STARTDESKTOP
[Desktop Entry]
Version=1.0
Type=Application
Name=Start HarpoonAI
Comment=Start HarpoonAI Enhanced System with Vector Embeddings
Exec=konsole --hold -e bash -c "cd '$PROJECT_DIR' && ./start_harpoonai.sh; echo ''; echo 'Press any key to close...'; read -n 1"
Icon=$ICONS_DIR/harpoonai-start.svg
Terminal=false
Categories=Development;Education;Science;
Keywords=AI;LLM;Chat;Vector;Embeddings;
StartupNotify=true
STARTDESKTOP

if [ ! -f "$APPLICATIONS_DIR/harpoonai-start.desktop" ]; then
    echo "❌ Failed to create Start shortcut"
else
    echo "✅ Start shortcut created"
fi

# Create Stop Desktop Entry
echo "📝 Creating Stop shortcut..."
cat > "$APPLICATIONS_DIR/harpoonai-stop.desktop" << STOPDESKTOP
[Desktop Entry]
Version=1.0
Type=Application
Name=Stop HarpoonAI
Comment=Stop HarpoonAI Enhanced System
Exec=konsole --hold -e bash -c "cd '$PROJECT_DIR' && ./stop_harpoonai.sh; echo ''; echo 'Press any key to close...'; read -n 1"
Icon=$ICONS_DIR/harpoonai-stop.svg
Terminal=false
Categories=Development;Education;Science;
Keywords=AI;LLM;Chat;Stop;
StartupNotify=true
STOPDESKTOP

if [ ! -f "$APPLICATIONS_DIR/harpoonai-stop.desktop" ]; then
    echo "❌ Failed to create Stop shortcut"
else
    echo "✅ Stop shortcut created"
fi

# Create Status Desktop Entry
echo "📝 Creating Status shortcut..."
cat > "$APPLICATIONS_DIR/harpoonai-status.desktop" << STATUSDESKTOP
[Desktop Entry]
Version=1.0
Type=Application
Name=HarpoonAI Status
Comment=Check HarpoonAI System Status
Exec=konsole --hold -e bash -c "cd '$PROJECT_DIR' && ./system_info.sh; echo ''; echo 'Press any key to close...'; read -n 1"
Icon=$ICONS_DIR/harpoonai-status.svg
Terminal=false
Categories=Development;Education;Science;
Keywords=AI;LLM;Status;Info;
StartupNotify=true
STATUSDESKTOP

if [ ! -f "$APPLICATIONS_DIR/harpoonai-status.desktop" ]; then
    echo "❌ Failed to create Status shortcut"
else
    echo "✅ Status shortcut created"
fi

# Create Open Browser Desktop Entry
echo "📝 Creating Browser shortcut..."
cat > "$APPLICATIONS_DIR/harpoonai-browser.desktop" << BROWSERDESKTOP
[Desktop Entry]
Version=1.0
Type=Application
Name=Open HarpoonAI
Comment=Open HarpoonAI Web Interface in Browser
Exec=xdg-open http://localhost:8000
Icon=$ICONS_DIR/harpoonai-browser.svg
Terminal=false
Categories=Development;Education;Science;
Keywords=AI;LLM;Chat;Browser;Web;
StartupNotify=true
BROWSERDESKTOP

if [ ! -f "$APPLICATIONS_DIR/harpoonai-browser.desktop" ]; then
    echo "❌ Failed to create Browser shortcut"
else
    echo "✅ Browser shortcut created"
fi

# Make desktop files executable
chmod +x "$APPLICATIONS_DIR/harpoonai-start.desktop"
chmod +x "$APPLICATIONS_DIR/harpoonai-stop.desktop"
chmod +x "$APPLICATIONS_DIR/harpoonai-status.desktop"
chmod +x "$APPLICATIONS_DIR/harpoonai-browser.desktop"

# Copy to Desktop if user wants desktop shortcuts
echo ""
echo "📋 Desktop shortcuts have been created in the applications menu."
echo ""
read -p "Do you also want shortcuts on your Desktop? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Ensure Desktop directory exists
    mkdir -p "$DESKTOP_DIR"
    
    # Copy desktop files to Desktop
    cp "$APPLICATIONS_DIR/harpoonai-start.desktop" "$DESKTOP_DIR/"
    cp "$APPLICATIONS_DIR/harpoonai-stop.desktop" "$DESKTOP_DIR/"
    cp "$APPLICATIONS_DIR/harpoonai-status.desktop" "$DESKTOP_DIR/"
    cp "$APPLICATIONS_DIR/harpoonai-browser.desktop" "$DESKTOP_DIR/"
    
    # Make them executable on Desktop too
    chmod +x "$DESKTOP_DIR/harpoonai-start.desktop"
    chmod +x "$DESKTOP_DIR/harpoonai-stop.desktop"
    chmod +x "$DESKTOP_DIR/harpoonai-status.desktop"
    chmod +x "$DESKTOP_DIR/harpoonai-browser.desktop"
    
    # Trust the desktop files (for KDE Plasma)
    gio set "$DESKTOP_DIR/harpoonai-start.desktop" "metadata::trusted" true 2>/dev/null || true
    gio set "$DESKTOP_DIR/harpoonai-stop.desktop" "metadata::trusted" true 2>/dev/null || true
    gio set "$DESKTOP_DIR/harpoonai-status.desktop" "metadata::trusted" true 2>/dev/null || true
    gio set "$DESKTOP_DIR/harpoonai-browser.desktop" "metadata::trusted" true 2>/dev/null || true
    
    echo "✅ Desktop shortcuts created on Desktop!"
fi

# Create a combined launcher script (optional)
cat > "$PROJECT_DIR/harpoonai-launcher.sh" << 'LAUNCHEREOF'
#!/bin/bash

# HarpoonAI Launcher Menu
PROJECT_DIR="$HOME/offline_ai_chat"
cd "$PROJECT_DIR"

while true; do
    clear
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "           🚀 HarpoonAI Enhanced Launcher"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  1) 🟢 Start HarpoonAI System"
    echo "  2) 🔴 Stop HarpoonAI System"
    echo "  3) 📊 Check System Status"
    echo "  4) 🌐 Open Web Interface"
    echo "  5) 📁 Open Project Directory"
    echo "  6) 📜 View Logs"
    echo "  7) 🔄 Restart System"
    echo "  0) ❌ Exit"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -n "Select option: "
    
    read choice
    
    case $choice in
        1)
            echo "Starting HarpoonAI..."
            ./start_harpoonai.sh
            echo "Press any key to continue..."
            read -n 1
            ;;
        2)
            echo "Stopping HarpoonAI..."
            ./stop_harpoonai.sh
            echo "Press any key to continue..."
            read -n 1
            ;;
        3)
            ./system_info.sh
            echo "Press any key to continue..."
            read -n 1
            ;;
        4)
            echo "Opening web interface..."
            xdg-open http://localhost:8000 &
            ;;
        5)
            echo "Opening project directory..."
            xdg-open "$PROJECT_DIR" &
            ;;
        6)
            echo "Select log to view:"
            echo "1) Backend log"
            echo "2) LLaMA log"
            echo "3) Falcon log"
            read -n 1 log_choice
            echo ""
            case $log_choice in
                1) less "$PROJECT_DIR/backend_server.log" ;;
                2) less "$PROJECT_DIR/llama_server.log" ;;
                3) less "$PROJECT_DIR/falcon_server.log" ;;
            esac
            ;;
        7)
            echo "Restarting HarpoonAI..."
            ./stop_harpoonai.sh
            sleep 2
            ./start_harpoonai.sh
            echo "Press any key to continue..."
            read -n 1
            ;;
        0)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid option!"
            sleep 1
            ;;
    esac
done
LAUNCHEREOF

chmod +x "$PROJECT_DIR/harpoonai-launcher.sh"

# Create launcher desktop entry
echo "📝 Creating Launcher shortcut..."
cat > "$APPLICATIONS_DIR/harpoonai-launcher.desktop" << LAUNCHERDESKTOP
[Desktop Entry]
Version=1.0
Type=Application
Name=HarpoonAI Launcher
Comment=HarpoonAI Control Center
Exec=konsole -e bash -c "cd '$PROJECT_DIR' && ./harpoonai-launcher.sh"
Icon=$ICONS_DIR/harpoonai-start.svg
Terminal=false
Categories=Development;Education;Science;
Keywords=AI;LLM;Chat;Launcher;Control;
StartupNotify=true
LAUNCHERDESKTOP

if [ ! -f "$APPLICATIONS_DIR/harpoonai-launcher.desktop" ]; then
    echo "❌ Failed to create Launcher shortcut"
else
    echo "✅ Launcher shortcut created"
    chmod +x "$APPLICATIONS_DIR/harpoonai-launcher.desktop"
fi

# Update KDE menu cache
update-desktop-database "$APPLICATIONS_DIR" 2>/dev/null || true

echo ""
echo "✅ Desktop shortcuts created successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📍 Shortcuts have been added to:"
echo "   • Application Menu (under Development/Education/Science)"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   • Desktop: $DESKTOP_DIR"
fi
echo ""
echo "🎯 Available shortcuts:"
echo "   • Start HarpoonAI - Start the system"
echo "   • Stop HarpoonAI - Stop the system"
echo "   • HarpoonAI Status - Check system status"
echo "   • Open HarpoonAI - Open web interface"
echo "   • HarpoonAI Launcher - Control center menu"
echo ""
echo "💡 Tips:"
echo "   • You can find the shortcuts in your Application Menu"
echo "   • Search for 'HarpoonAI' in the application launcher"
echo "   • You can pin these to your taskbar for quick access"
echo "   • Right-click on desktop shortcuts and select 'Trust' if needed"
echo ""
echo "📂 File locations:"
echo "   • Desktop files: $APPLICATIONS_DIR"
echo "   • Icons: $ICONS_DIR"
echo "   • Project: $PROJECT_DIR"
echo ""
echo "🔧 Troubleshooting:"
echo "   • If shortcuts don't appear, run: update-desktop-database"
echo "   • For KDE, logout and login again to refresh menu"
echo "   • To trust desktop icons: gio set [file] metadata::trusted true"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
