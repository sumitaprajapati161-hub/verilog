import os, re, html, shutil, subprocess, traceback
from pathlib import Path
from typing import List, Optional, TypedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

BASE = Path(__file__).resolve().parent
OUT = BASE / 'outputs'
OUT.mkdir(exist_ok=True)
app = FastAPI(title='Digital Logic Verilog AI', version='1.0.0')
app.mount('/outputs', StaticFiles(directory=str(OUT)), name='outputs')

API_KEY = os.getenv('GEMINI_API_KEY')
MODEL = os.getenv('GEMINI_MODEL', 'gemini-3.1-flash-lite-preview')
llm = ChatGoogleGenerativeAI(model=MODEL, google_api_key=API_KEY, temperature=0) if API_KEY else None

class CrewState(TypedDict, total=False):
    messages: List[BaseMessage]; next_step: Optional[str]; task: Optional[str]
    design_description: Optional[str]; gate_level_code: Optional[str]
    dataflow_level_code: Optional[str]; behavioral_level_code: Optional[str]
    testbench: Optional[str]; truth_table: Optional[str]
    simulation_result: Optional[str]; report: Optional[str]
    waveform_file: Optional[str]; circuit_file: Optional[str]
    truth_table_file: Optional[str]; warnings: List[str]

def clean_verilog_code(code):
    if not code: return ''
    return str(code).replace('```verilog','').replace('```Verilog','').replace('```VERILOG','').replace('```','').strip()

def extract_section(text, start, end):
    if not text or start not in text: return ''
    s = text.find(start) + len(start); e = text.find(end, s)
    return text[s:] .strip() if e == -1 else text[s:e].strip()

def extract_code(text):
    text = clean_verilog_code(text)
    p = text.find('module')
    return text[p:].strip() if p >= 0 else text

def validate_verilog_module(code, expected):
    if not code: return False, 'Code is empty.'
    if 'module' not in code: return False, 'module declaration is missing.'
    if 'endmodule' not in code: return False, 'endmodule is missing.'
    if expected not in code: return False, f"Expected module '{expected}' not found."
    return True, 'Complete module found.'

def write_file(path, content): Path(path).write_text(content, encoding='utf-8')

# ---------------- AOI21 ----------------
def generate_aoi21_design():
    description = '''AOI21 (AND-OR-INVERT) logic combines an AND operation and an OR operation followed by an inverter.

Boolean expression:
Y = ~((A & B) | C)

Operation:
1. A and B are connected to an AND gate.
2. The AND output is ORed with C.
3. The OR output is inverted.
4. The final output is Y.'''
    gate = '''`timescale 1ns/1ps
module gate_level_model (input wire A, input wire B, input wire C, output wire Y);
    wire AB;
    wire OR_OUT;
    and G1(AB, A, B);
    or  G2(OR_OUT, AB, C);
    not G3(Y, OR_OUT);
endmodule'''
    data = '''`timescale 1ns/1ps
module dataflow_model (input wire A, input wire B, input wire C, output wire Y);
    assign Y = ~((A & B) | C);
endmodule'''
    beh = '''`timescale 1ns/1ps
module behavioral_model (input wire A, input wire B, input wire C, output reg Y);
    always @(*) begin
        if (((A & B) | C) == 1'b1) Y = 1'b0;
        else Y = 1'b1;
    end
endmodule'''
    return description, gate, data, beh

def generate_aoi21_testbench():
    return '''`timescale 1ns/1ps
module testbench;
    reg A, B, C;
    wire Y_GATE, Y_DATA, Y_BEHAV;
    gate_level_model DUT_GATE(.A(A),.B(B),.C(C),.Y(Y_GATE));
    dataflow_model DUT_DATA(.A(A),.B(B),.C(C),.Y(Y_DATA));
    behavioral_model DUT_BEHAV(.A(A),.B(B),.C(C),.Y(Y_BEHAV));
    initial begin $dumpfile("aoi21_waveform.vcd"); $dumpvars(0,testbench); end
    initial begin
        A=0; B=0; C=0; #10; $display("TIME=%0t A=%b B=%b C=%b Y_GATE=%b Y_DATA=%b Y_BEHAV=%b",$time,A,B,C,Y_GATE,Y_DATA,Y_BEHAV);
        A=0; B=0; C=1; #10; $display("TIME=%0t A=%b B=%b C=%b Y_GATE=%b Y_DATA=%b Y_BEHAV=%b",$time,A,B,C,Y_GATE,Y_DATA,Y_BEHAV);
        A=0; B=1; C=0; #10; $display("TIME=%0t A=%b B=%b C=%b Y_GATE=%b Y_DATA=%b Y_BEHAV=%b",$time,A,B,C,Y_GATE,Y_DATA,Y_BEHAV);
        A=0; B=1; C=1; #10; $display("TIME=%0t A=%b B=%b C=%b Y_GATE=%b Y_DATA=%b Y_BEHAV=%b",$time,A,B,C,Y_GATE,Y_DATA,Y_BEHAV);
        A=1; B=0; C=0; #10; $display("TIME=%0t A=%b B=%b C=%b Y_GATE=%b Y_DATA=%b Y_BEHAV=%b",$time,A,B,C,Y_GATE,Y_DATA,Y_BEHAV);
        A=1; B=0; C=1; #10; $display("TIME=%0t A=%b B=%b C=%b Y_GATE=%b Y_DATA=%b Y_BEHAV=%b",$time,A,B,C,Y_GATE,Y_DATA,Y_BEHAV);
        A=1; B=1; C=0; #10; $display("TIME=%0t A=%b B=%b C=%b Y_GATE=%b Y_DATA=%b Y_BEHAV=%b",$time,A,B,C,Y_GATE,Y_DATA,Y_BEHAV);
        A=1; B=1; C=1; #10; $display("TIME=%0t A=%b B=%b C=%b Y_GATE=%b Y_DATA=%b Y_BEHAV=%b",$time,A,B,C,Y_GATE,Y_DATA,Y_BEHAV);
        $display("SIMULATION COMPLETED"); $finish;
    end
endmodule'''

def generate_aoi_truth_table():
    rows = [f'| {a} | {b} | {c} | {int(not ((a and b) or c))} |' for a in [0,1] for b in [0,1] for c in [0,1]]
    return '| A | B | C | Y |\n|---|---|---|---|\n' + '\n'.join(rows)

def create_truth_table_image():
    fig, ax = plt.subplots(figsize=(6,4)); ax.axis('off')
    data=[['0','0','0','1'],['0','0','1','0'],['0','1','0','1'],['0','1','1','0'],['1','0','0','1'],['1','0','1','0'],['1','1','0','0'],['1','1','1','0']]
    t=ax.table(cellText=data,colLabels=['A','B','C','Y'],loc='center',cellLoc='center'); t.auto_set_font_size(False); t.set_fontsize(13); t.scale(1.2,1.7)
    ax.set_title('AOI21 Truth Table\nY = ~((A & B) | C)')
    f=OUT/'aoi21_truth_table.png'; plt.savefig(f,dpi=200,bbox_inches='tight'); plt.close(); return f

def create_aoi_waveform():
    combos=[(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]
    t=[]; vals=[[],[],[],[]]
    for i,(a,b,c) in enumerate(combos):
        y=int(not ((a and b) or c)); t += [i*10,(i+1)*10]
        for arr,v in zip(vals,[a,b,c,y]): arr += [v,v]
    fig,ax=plt.subplots(figsize=(12,6))
    for arr,label in zip(vals,['A','B','C','Y']): ax.step(t,arr,where='post',label=label)
    ax.set(xlabel='Time (ns)',ylabel='Logic Level',title='AOI21 Timing Waveform'); ax.set_yticks([0,1]); ax.set_xticks(range(0,81,10)); ax.grid(True); ax.legend()
    f=OUT/'aoi21_timing_waveform.png'; plt.savefig(f,dpi=200,bbox_inches='tight'); plt.close(); return f

def create_aoi_circuit_diagram():
    fig,ax=plt.subplots(figsize=(14,6)); ax.set_xlim(0,14); ax.set_ylim(0,8); ax.axis('off')
    ax.text(7,7.5,'AOI21 AND-OR-INVERT LOGIC CIRCUIT',ha='center',fontsize=18,fontweight='bold')
    boxes=[(3,4.5,2,1.8,'AND\nGATE'),(6.5,4.5,2,1.8,'OR\nGATE'),(10,4.5,1.5,1.8,'NOT\nGATE')]
    for x,y,w,h,label in boxes:
        ax.add_patch(patches.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05',linewidth=2,fill=False)); ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=13,fontweight='bold')
    ax.annotate('A',xy=(3,5.75),xytext=(.8,5.75),arrowprops=dict(arrowstyle='->',linewidth=2),fontsize=14,fontweight='bold')
    ax.annotate('B',xy=(3,5.05),xytext=(.8,5.05),arrowprops=dict(arrowstyle='->',linewidth=2),fontsize=14,fontweight='bold')
    ax.annotate('',xy=(6.5,5.4),xytext=(5,5.4),arrowprops=dict(arrowstyle='->',linewidth=2)); ax.text(5.7,6,'AB = A · B',ha='center')
    ax.annotate('C',xy=(6.5,4.85),xytext=(5,2.5),arrowprops=dict(arrowstyle='->',linewidth=2,connectionstyle='angle3'),fontsize=14,fontweight='bold')
    ax.annotate('',xy=(10,5.4),xytext=(8.5,5.4),arrowprops=dict(arrowstyle='->',linewidth=2)); ax.text(9.25,6,'AB + C',ha='center')
    ax.annotate('Y = ~((A · B) + C)',xy=(11.5,5.4),xytext=(12,5.4),arrowprops=dict(arrowstyle='->',linewidth=2),fontsize=13,fontweight='bold',va='center')
    ax.text(7,1,'Boolean Expression: Y = ~((A & B) | C)',ha='center',fontsize=15,fontweight='bold')
    f=OUT/'aoi21_circuit_diagram.png'; plt.savefig(f,dpi=250,bbox_inches='tight'); plt.close(); return f

# ---------------- Generic Gemini ----------------
def generate_generic_design(task):
    if not llm: raise RuntimeError('GEMINI_API_KEY is not configured.')
    prompt = f'''You are a senior Verilog RTL and digital logic design engineer.
USER TASK: {task}

Return ONLY these sections:
===DESIGN_DESCRIPTION===
===BOOLEAN_EXPRESSION===
===GATE_LEVEL_CODE===
===DATAFLOW_CODE===
===BEHAVIORAL_CODE===
===TESTBENCH===
===TRUTH_TABLE===
===END===

Give three COMPLETE, standalone Verilog-2001 modules named exactly gate_level_model, dataflow_model and behavioral_model. Use the same ports in all three. Include module declaration, ports, implementation and endmodule. Include a complete testbench that instantiates all three and tests meaningful combinations. Do not use SystemVerilog or pseudocode.'''
    r=llm.invoke(prompt); c=r.content
    if isinstance(c,list): c='\n'.join(i.get('text','') if isinstance(i,dict) else str(i) for i in c)
    return str(c)

def simulate(gate,data,beh,tb):
    iv=shutil.which('iverilog'); vv=shutil.which('vvp')
    if not iv or not vv: return 'VERILOG SIMULATION UNAVAILABLE\n\nIcarus Verilog is not installed on this server.'
    work=OUT/'simulation'; work.mkdir(exist_ok=True)
    design=work/'all_models.v'; test=work/'testbench.v'; exe=work/'simulation.out'
    write_file(design,gate+'\n\n'+data+'\n\n'+beh); write_file(test,tb)
    cp=subprocess.run([iv,'-o',str(exe),str(design),str(test)],capture_output=True,text=True,timeout=30)
    if cp.returncode: return 'VERILOG COMPILATION ERROR\n\n'+cp.stderr
    sp=subprocess.run([vv,str(exe)],capture_output=True,text=True,cwd=str(work),timeout=30)
    if sp.returncode: return 'VERILOG SIMULATION ERROR\n\n'+sp.stderr
    vcd=work/'aoi21_waveform.vcd'
    if vcd.exists(): shutil.copy2(vcd,OUT/'aoi21_waveform.vcd')
    return 'VERILOG COMPILATION: SUCCESS\nVERILOG SIMULATION: SUCCESS\n\nSIMULATION OUTPUT:\n\n'+sp.stdout

def develop(task):
    if 'aoi' in task.lower() and ('logic' in task.lower() or 'gate' in task.lower()):
        d,g,df,b=generate_aoi21_design(); return {'task':task,'design_description':d,'gate_level_code':g,'dataflow_level_code':df,'behavioral_level_code':b,'warnings':[]}
    r=generate_generic_design(task)
    desc=extract_section(r,'===DESIGN_DESCRIPTION===','===BOOLEAN_EXPRESSION===')
    gate=extract_code(extract_section(r,'===GATE_LEVEL_CODE===','===DATAFLOW_CODE==='))
    data=extract_code(extract_section(r,'===DATAFLOW_CODE===','===BEHAVIORAL_CODE==='))
    beh=extract_code(extract_section(r,'===BEHAVIORAL_CODE===','===TESTBENCH==='))
    tb=clean_verilog_code(extract_section(r,'===TESTBENCH===','===TRUTH_TABLE==='))
    truth=extract_section(r,'===TRUTH_TABLE===','===END===')
    warnings=[]
    for code,name in [(gate,'gate_level_model'),(data,'dataflow_model'),(beh,'behavioral_model')]:
        ok,msg=validate_verilog_module(code,name)
        if not ok: warnings.append(msg)
    return {'task':task,'design_description':desc,'gate_level_code':gate,'dataflow_level_code':data,'behavioral_level_code':beh,'testbench':tb,'truth_table':truth,'warnings':warnings}

def test(state):
    task=state['task']; aoi='aoi' in task.lower()
    tb=generate_aoi21_testbench() if aoi else state.get('testbench','')
    truth=generate_aoi_truth_table() if aoi else state.get('truth_table','')
    if not tb:
        if not llm: raise RuntimeError('GEMINI_API_KEY is not configured.')
        r=llm.invoke(f'''Generate one complete Verilog-2001 testbench for this task: {task}. Instantiate gate_level_model, dataflow_model and behavioral_model, test meaningful combinations, compare outputs, use $dumpfile, $dumpvars and $finish, and return only code.''')
        tb=clean_verilog_code(r.content if isinstance(r.content,str) else ''.join(str(x) for x in r.content))
    sim=simulate(state['gate_level_code'],state['dataflow_level_code'],state['behavioral_level_code'],tb)
    wf=cd=tt=None
    if aoi:
        wf=create_aoi_waveform(); cd=create_aoi_circuit_diagram(); tt=create_truth_table_image()
    report=f'''============================================================
        DIGITAL LOGIC VERILOG DESIGN REPORT
============================================================

TASK
------------------------------------------------------------
{task}

============================================================
1. DESIGN DESCRIPTION
============================================================

{state.get('design_description','')}

============================================================
2. GATE-LEVEL MODELLING
   COMPLETE VERILOG CODE
============================================================

{state['gate_level_code']}

============================================================
3. DATA-FLOW MODELLING
   COMPLETE VERILOG CODE
============================================================

{state['dataflow_level_code']}

============================================================
4. BEHAVIORAL-LEVEL MODELLING
   COMPLETE VERILOG CODE
============================================================

{state['behavioral_level_code']}

============================================================
5. COMPLETE TESTBENCH
============================================================

{tb}

============================================================
6. TRUTH TABLE
============================================================

{truth}

============================================================
7. SIMULATION RESULT
============================================================

{sim}

============================================================
                 END OF REPORT
============================================================'''
    return {**state,'testbench':tb,'truth_table':truth,'simulation_result':sim,'report':report,'waveform_file':str(wf) if wf else None,'circuit_file':str(cd) if cd else None,'truth_table_file':str(tt) if tt else None}

def img_block(title,path):
    if not path: return ''
    name=Path(path).name
    return f'<div class="section"><h2>{html.escape(title)}</h2><img src="/outputs/{html.escape(name)}" alt="{html.escape(title)}"></div>' if (OUT/name).exists() else ''

def code_block(title,code):
    return f'<div class="section"><h2>{html.escape(title)}</h2><pre>{html.escape(code or "")}</pre></div>'

def page(state):
    w=''.join(f'<li>{html.escape(x)}</li>' for x in state.get('warnings',[]))
    warnings=f'<div class="warning"><b>Warnings</b><ul>{w}</ul></div>' if w else ''
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Digital Logic Verilog AI</title><style>
body{{font-family:Arial;background:#f4f6f8;margin:0;padding:25px}}.container{{max-width:1200px;margin:auto}}.header,.section{{background:white;border-radius:10px;padding:20px;margin:15px 0}}.header{{background:#1f2937;color:white}}pre{{background:#111827;color:#f9fafb;padding:18px;border-radius:8px;overflow:auto;white-space:pre-wrap}}img{{max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px}}input{{width:75%;padding:13px;font-size:16px;border-radius:7px;border:1px solid #bbb}}button{{padding:13px 20px;background:#2563eb;color:white;border:0;border-radius:7px}}.warning{{background:#fff3cd;padding:15px;border-radius:8px}}</style></head><body><div class="container"><div class="header"><h1>Digital Logic Verilog AI</h1><p>Complete design, simulation and visual output</p></div><form method="post" action="/generate"><input name="task" value="{html.escape(state['task'])}" required><button>Generate</button></form>{warnings}<div class="section"><h2>Task</h2><p>{html.escape(state['task'])}</p></div><div class="section"><h2>1. Design Description</h2><pre>{html.escape(state.get('design_description',''))}</pre></div>{code_block('2. Gate-Level Modelling - Complete Verilog Code',state.get('gate_level_code'))}{code_block('3. Data-Flow Modelling - Complete Verilog Code',state.get('dataflow_level_code'))}{code_block('4. Behavioral-Level Modelling - Complete Verilog Code',state.get('behavioral_level_code'))}{code_block('5. Complete Testbench',state.get('testbench'))}<div class="section"><h2>6. Truth Table</h2><pre>{html.escape(state.get('truth_table',''))}</pre></div><div class="section"><h2>7. Simulation Result</h2><pre>{html.escape(state.get('simulation_result',''))}</pre></div>{img_block('8. Timing Waveform',state.get('waveform_file'))}{img_block('9. Complete Circuit Diagram',state.get('circuit_file'))}{img_block('10. Truth Table Image',state.get('truth_table_file'))}</div></body></html>'''

@app.get('/',response_class=HTMLResponse)
def home():
    return '''<!doctype html><html><head><meta charset="utf-8"><title>Digital Logic Verilog AI</title><style>body{font-family:Arial;background:#f4f6f8;padding:40px}.box{max-width:800px;margin:auto;background:white;padding:30px;border-radius:12px}input{width:75%;padding:14px;font-size:16px}button{padding:14px 22px;background:#2563eb;color:white;border:0;border-radius:7px}</style></head><body><div class="box"><h1>Digital Logic Verilog AI</h1><p>Enter a task such as <b>give an AOI logic</b>.</p><form method="post" action="/generate"><input name="task" placeholder="give an AOI logic" required><button>Generate</button></form></div></body></html>'''

@app.post('/generate',response_class=HTMLResponse)
def generate(task:str=Form(...)):
    try: return HTMLResponse(page(test(develop(task.strip()))))
    except Exception as e: return HTMLResponse(f'<h1>Error</h1><pre>{html.escape(traceback.format_exc())}</pre>',status_code=500)

@app.get('/health')
def health(): return {'status':'ok','gemini_configured':bool(API_KEY),'iverilog_installed':bool(shutil.which('iverilog'))}

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='0.0.0.0',port=int(os.getenv('PORT','8000')))
