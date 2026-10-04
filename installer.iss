; installer.iss — Inno Setup script for the Windows installer.
;
; Compiled by .github/workflows/release.yml after the PyInstaller build:
;     ISCC.exe installer.iss        (LOREBOOK_VERSION env var must be set)
; Input:  dist\LoreBook\  (PyInstaller onedir output)
; Output: Output\LoreBook-Setup-<version>.exe
;
; Per-user install (PrivilegesRequired=lowest): no UAC prompt; {autopf}
; resolves to %LOCALAPPDATA%\Programs. App data (settings, card images,
; caches, CSVs, logs) lives separately in %LOCALAPPDATA%\LoreBook and
; deliberately survives uninstall — see lorebook/core/paths.py.

#define MyAppName "Lore Book"
#define MyAppVersion GetEnv("LOREBOOK_VERSION")
#define MyAppExeName "LoreBook.exe"

[Setup]
; Fixed GUID: never change it, or upgrades stop installing over the top.
AppId={{7AD29961-4C33-4BC7-A82B-A40558BC3ACD}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Lore-Book
AppPublisherURL=https://github.com/Isaac-de-Leon/Lore-Book
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputBaseFilename=LoreBook-Setup-{#MyAppVersion}
SetupIconFile=lorebook\ui\assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; Flags: unchecked

[Files]
Source: "dist\LoreBook\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
