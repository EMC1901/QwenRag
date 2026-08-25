; Offline, per-user QwenRAG installer. Build with Inno Setup 6.x x64.

#ifndef RuntimeDir
  #define RuntimeDir AddBackslash(SourcePath) + "..\build\dist\QwenRagRuntime"
#endif
#ifndef OcrDir
  #define OcrDir AddBackslash(SourcePath) + "..\..\models\ocr"
#endif
#ifndef InitialKbDir
  #define InitialKbDir AddBackslash(SourcePath) + "..\payload\initial_kb"
#endif
#ifexist InitialKbDir + "\SHA256SUMS.txt"
  #define HasInitialKnowledgeBase 1
#else
  #define HasInitialKnowledgeBase 0
#endif
#ifndef MinimumFreeSpaceWithoutKbMB
  ; Includes runtime extraction and the first empty knowledge base.  Release
  ; engineering may raise this via ISCC /D after measuring a larger runtime.
  #define MinimumFreeSpaceWithoutKbMB 2048
#endif
#ifndef MinimumFreeSpaceWithKbMB
  ; The release build computes a payload-specific value.  This fallback keeps
  ; sufficient headroom if the script is invoked directly.
  #define MinimumFreeSpaceWithKbMB 12288
#endif

[Setup]
AppId={{534F9D3C-8770-4F40-91C3-DB472AEB9342}
AppName=QwenRAG
AppVersion=1.0.0
AppPublisher=QwenRAG
DefaultDirName={localappdata}\Programs\QwenRAG
DefaultGroupName=QwenRAG
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2/ultra64
SolidCompression=yes
DiskSpanning=yes
DiskSliceSize=2000000000
OutputDir=..\output
OutputBaseFilename=QwenRAG-1.0.0-Setup
UninstallDisplayName=QwenRAG
DisableProgramGroupPage=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Types]
Name: "full"; Description: "完整安装（含随包初始知识库）"
Name: "custom"; Description: "自定义安装"

[Components]
Name: "core"; Description: "QwenRAG 核心程序"; Types: full custom; Flags: fixed
#if HasInitialKnowledgeBase
Name: "initial_kb"; Description: "安装随包初始知识库"; Types: full
#endif

[Files]
Source: "{#RuntimeDir}\*"; DestDir: "{app}"; Components: core; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "{#OcrDir}\*"; DestDir: "{app}\resources\ocr"; Components: core; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\..\launch\Start-QwenRAG.ps1"; DestDir: "{app}\launch"; Components: core; Flags: ignoreversion
Source: "..\..\scripts\submit_incremental_import.ps1"; DestDir: "{app}\scripts"; Components: core; Flags: ignoreversion
Source: "..\..\scripts\submit_incremental_import.bat"; DestDir: "{app}\scripts"; Components: core; Flags: ignoreversion
#if HasInitialKnowledgeBase
; SQLite and FAISS assets are already binary-dense.  Avoiding LZMA recompression
; makes large customer knowledge-base media build and install predictably while
; SHA-256 verification still protects every copied file.
Source: "{#InitialKbDir}\*"; DestDir: "{localappdata}\QwenRAG\data"; Components: initial_kb; Check: ShouldInstallInitialKnowledgeBase; Flags: recursesubdirs createallsubdirs ignoreversion nocompression
#endif

[Icons]
Name: "{group}\启动 QwenRAG"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\launch\Start-QwenRAG.ps1"""; WorkingDir: "{app}"
Name: "{group}\资料入库工作台"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\submit_incremental_import.ps1"" -WaitForCompletion -KeepWindowOpen"; WorkingDir: "{localappdata}\QwenRAG\data\workbench"
Name: "{group}\配置与诊断工具"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -Command ""& ''{app}\QwenRagRuntime.exe'' config validate; Read-Host ''Press Enter to close''"""; WorkingDir: "{app}"
Name: "{group}\卸载 QwenRAG"; Filename: "{uninstallexe}"
Name: "{autodesktop}\启动 QwenRAG"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\launch\Start-QwenRAG.ps1"""; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{autodesktop}\资料入库工作台"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\submit_incremental_import.ps1"" -WaitForCompletion -KeepWindowOpen"; WorkingDir: "{localappdata}\QwenRAG\data\workbench"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; Flags: unchecked

[UninstallDelete]
; Customer data is intentionally not listed here. Uninstall keeps
; %LOCALAPPDATA%\QwenRAG\config, data, archive, results and logs intact.
Type: filesandordirs; Name: "{app}"

[Code]
const
  KbMissing = 0;
  KbValid = 1;
  KbInvalid = 2;

var
  ExistingKbState: Integer;

function DataRoot: String;
begin
  Result := ExpandConstant('{localappdata}\QwenRAG\data');
end;

function KnowledgeBaseState: Integer;
var
  Root: String;
begin
  Root := DataRoot;
  if not DirExists(Root) then begin
    Result := KbMissing;
    exit;
  end;
  if FileExists(AddBackslash(Root) + 'metadata.db') and
     FileExists(AddBackslash(Root) + 'knowledge_manifest.json') and
     FileExists(AddBackslash(Root) + 'vector_index\index.faiss') and
     FileExists(AddBackslash(Root) + 'vector_index\index.meta.json') then begin
    Result := KbValid;
  end else begin
    Result := KbInvalid;
  end;
end;

function InitializeSetup: Boolean;
var
  ExistingRuntime: String;
  ResultCode: Integer;
begin
  ExistingRuntime := ExpandConstant('{localappdata}\Programs\QwenRAG\QwenRagRuntime.exe');
  if FileExists(ExistingRuntime) then begin
    if Exec(ExistingRuntime, 'check-runtime-active', ExpandConstant('{localappdata}\Programs\QwenRAG'), SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 10) then begin
      MsgBox('QwenRAG 或资料入库任务正在运行。请先关闭启动或入库窗口后再安装、升级或卸载。', mbError, MB_OK);
      Result := False;
      exit;
    end;
  end;
  ExistingKbState := KnowledgeBaseState;
  if ExistingKbState = KbInvalid then begin
    MsgBox('检测到已有但不完整的 QwenRAG 知识库。安装器不会覆盖它。请先备份 %LOCALAPPDATA%\QwenRAG 后联系技术支持。', mbError, MB_OK);
    Result := False;
    exit;
  end;
  Result := True;
end;

function RequiredFreeBytes: Int64;
begin
  Result := Int64({#MinimumFreeSpaceWithoutKbMB}) * 1024 * 1024;
#if HasInitialKnowledgeBase
  if IsComponentSelected('initial_kb') then begin
    Result := Int64({#MinimumFreeSpaceWithKbMB}) * 1024 * 1024;
  end;
#endif
end;

function CheckInstallDiskSpace: Boolean;
var
  FreeBytes, TotalBytes, RequiredBytes: Int64;
begin
  RequiredBytes := RequiredFreeBytes;
  if not GetSpaceOnDisk64(ExpandConstant('{localappdata}'), FreeBytes, TotalBytes) then begin
    MsgBox('无法读取 %LOCALAPPDATA% 所在磁盘的可用空间，安装已停止。', mbError, MB_OK);
    Result := False;
    exit;
  end;
  if FreeBytes < RequiredBytes then begin
    MsgBox(
      '磁盘可用空间不足。请选择空间更充足的用户配置目录，或释放磁盘空间后重试。' + #13#10 +
      '当前至少需要 ' + IntToStr(RequiredBytes div 1024 div 1024) + ' MB 可用空间。',
      mbError, MB_OK);
    Result := False;
    exit;
  end;
  Result := True;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = wpReady then begin
    Result := CheckInstallDiskSpace;
  end;
end;

function ShouldInstallInitialKnowledgeBase: Boolean;
begin
#if HasInitialKnowledgeBase
  Result := (ExistingKbState = KbMissing) and IsComponentSelected('initial_kb');
#else
  Result := False;
#endif
end;

function VerifyInitialKnowledgeBase: Boolean;
var
  Lines: TArrayOfString;
  I, SplitAt: Integer;
  ExpectedHash, RelativeName, ActualHash, FullName: String;
begin
  Result := True;
#if HasInitialKnowledgeBase
  if not ShouldInstallInitialKnowledgeBase then exit;
  if not LoadStringsFromFile(AddBackslash(DataRoot) + 'SHA256SUMS.txt', Lines) then begin
    Result := False;
    exit;
  end;
  for I := 0 to GetArrayLength(Lines) - 1 do begin
    SplitAt := Pos('  ', Lines[I]);
    if SplitAt <= 0 then begin Result := False; exit; end;
    ExpectedHash := Copy(Lines[I], 1, SplitAt - 1);
    RelativeName := Copy(Lines[I], SplitAt + 2, MaxInt);
    StringChangeEx(RelativeName, '/', '\', True);
    FullName := AddBackslash(DataRoot) + RelativeName;
    if not FileExists(FullName) then begin Result := False; exit; end;
    ActualHash := GetSHA256OfFile(FullName);
    if CompareText(ExpectedHash, ActualHash) <> 0 then begin Result := False; exit; end;
  end;
#endif
end;

function RunRuntime(const Parameters: String): Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec(AddBackslash(ExpandConstant('{app}')) + 'QwenRagRuntime.exe', Parameters, ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep <> ssPostInstall then exit;
  if not VerifyInitialKnowledgeBase then begin
    DelTree(DataRoot, True, True, True);
    MsgBox('初始知识库完整性校验失败，安装已停止。请重新获取完整安装介质。', mbError, MB_OK);
    RaiseException('Initial knowledge-base integrity verification failed.');
  end;
  if not RunRuntime('config migrate') then begin
    MsgBox('无法备份或迁移现有 QwenRAG 配置。原配置已保留，升级已停止。', mbError, MB_OK);
    RaiseException('QwenRAG configuration migration failed.');
  end;
  if not RunRuntime('config init') then begin
    MsgBox('无法初始化 QwenRAG 配置。请查看 %LOCALAPPDATA%\QwenRAG\logs。', mbError, MB_OK);
    RaiseException('QwenRAG configuration initialization failed.');
  end;
  if (ExistingKbState = KbMissing) and not ShouldInstallInitialKnowledgeBase then begin
    if not RunRuntime('kb-init-empty') then begin
      MsgBox('无法创建空知识库。请查看 %LOCALAPPDATA%\QwenRAG\logs。', mbError, MB_OK);
      RaiseException('Empty knowledge-base initialization failed.');
    end;
  end;
  if not RunRuntime('diagnose-install') then begin
    MsgBox('安装诊断未通过。请查看 %LOCALAPPDATA%\QwenRAG\logs 并联系技术支持。', mbError, MB_OK);
    RaiseException('QwenRAG install diagnosis failed.');
  end;
  MsgBox('QwenRAG 已安装。模型与推理服务需由实施人员按交付说明离线配置；安装器不会联网下载任何内容。', mbInformation, MB_OK);
end;
