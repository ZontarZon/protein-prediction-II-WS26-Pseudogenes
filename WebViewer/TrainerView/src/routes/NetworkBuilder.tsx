import { useEffect, useRef, useState } from 'react';
import { get, post } from './api';
import './NetworkBuilder.css';

type Params = Record<string, unknown>;
type Block = {id: string; type: string; params: Params; inputs: string[]; x: number; y: number};
type Graph = {version: number; name: string; nodes: Block[]};
type Item = {name: string; group: string; defaults: Params; required: string[]};
type Metrics = {inferred: Record<string, Params>; parameters: number; weight_bytes: number; buffer_bytes: number; activation_bytes: number; inference_bytes: number; training_bytes: number; nodes: Record<string, {shape: number[]; dtype: string; bytes: number}>};
const initial = (): Graph => ({version: 1, name: 'MyNetwork', nodes: [{id:'input', type:'Input', params:{shape:[1,4], dtype:'float32'}, inputs:[], x:60, y:150}]});
const bytes = (n: number) => n >= 1048576 ? `${(n/1048576).toFixed(2)} MiB` : `${(n/1024).toFixed(2)} KiB`;
export function NetworkBuilder() {
  const [graph, setGraph] = useState<Graph>(() => {try { const saved = localStorage.getItem('network-builder-v1'); return saved ? JSON.parse(saved) : initial(); } catch {return initial();}});
  const [catalog, setCatalog] = useState<Item[]>([]), [version, setVersion] = useState('');
  const [group, setGroup] = useState('Linear & embedding'), [search, setSearch] = useState('');
  const [selected, setSelected] = useState('input'), [connecting, setConnecting] = useState('');
  const [right, setRight] = useState(true), [toolbar, setToolbar] = useState(true);
  const [camera, setCamera] = useState({x:0,y:0,z:1});
  const [metrics, setMetrics] = useState<Metrics>(), [error, setError] = useState(''), [notice, setNotice] = useState('');
  const [models, setModels] = useState<string[]>([]), [importModel, setImportModel] = useState(''), [classes, setClasses] = useState<string[]>([]);
  const [busy, setBusy] = useState(false), [fieldError, setFieldError] = useState('');
  const [history, setHistory] = useState<Graph[]>([]), [future, setFuture] = useState<Graph[]>([]);
  const canvas = useRef<HTMLDivElement>(null);
  const graphFile = useRef<HTMLInputElement>(null);
  const drag = useRef<{id?: string; x:number; y:number; ox:number; oy:number} | null>(null);
  const node = graph.nodes.find(n => n.id === selected);
  function change(next: Graph) {setHistory(h=>[...h,graph]); setFuture([]); setGraph(next); setNotice('');}
  function editNode(patch: Partial<Block>) {if (node) change({...graph,nodes:graph.nodes.map(n=>n.id===node.id?{...n,...patch}:n)});}
  useEffect(()=>{get<{items:Item[];version:string}>('/network-builder/catalog').then(r=>{setCatalog(r.items);setVersion(r.version);}).catch(e=>setError(e.message));get<{models:string[]}>('/network-builder/models').then(r=>setModels(r.models)).catch(e=>setError(e.message));},[]);
  useEffect(()=>{localStorage.setItem('network-builder-v1',JSON.stringify(graph)); let alive=true; const timer=setTimeout(()=>{post<Metrics>('/network-builder/analyze',graph).then(r=>{if(alive){setMetrics(r);setError('');setGraph(g=>{const nodes=g.nodes.map(n=>r.inferred[n.id]?{...n,params:r.inferred[n.id]}:n);return JSON.stringify(nodes)===JSON.stringify(g.nodes)?g:{...g,nodes};});}}).catch(e=>{if(alive){setMetrics(undefined);setError(e.message);}});},350);return()=>{alive=false;clearTimeout(timer);};},[graph]);
  function add(type:string,x?:number,y?:number) {
    const entry=catalog.find(c=>c.name===type);
    const id=`n_${crypto.randomUUID().replaceAll("-", "_")}`;
    const b=canvas.current?.getBoundingClientRect();
    const params=type==='Input'?{shape:[1,4],dtype:'float32'}:structuredClone(entry?.defaults??{});
    change({...graph,nodes:[...graph.nodes,{id,type,params,inputs:[],x:x??((b?.width??800)/2-camera.x)/camera.z,y:y??(200-camera.y)/camera.z}]});setSelected(id);setRight(true);
  }
  function connect(target:string) {
    if(!connecting || connecting===target)return;
    const targetNode=graph.nodes.find(n=>n.id===target)!;
    if(targetNode.type==='Input')return;
    const downstream=(id:string,seen=new Set<string>()):boolean=>{if(id===target)return true;if(seen.has(id))return false;seen.add(id);return (graph.nodes.find(n=>n.id===id)?.inputs??[]).some(i=>downstream(i,seen));};
    if(downstream(connecting)){setNotice('That connection would create a cycle.');setConnecting('');return;}
    const multi=['Add','Multiply','Concat','Bilinear','MultiheadAttention'].includes(targetNode.type);
    const inputs=multi?[...new Set([...targetNode.inputs,connecting])]:[connecting];
    const shape=metrics?.nodes[connecting]?.shape;
    const params={...targetNode.params};
    if(shape){if('in_features' in params)params.in_features=shape[shape.length-1];if('in_channels' in params)params.in_channels=shape[1];if('num_features' in params)params.num_features=shape[1];if('input_size' in params)params.input_size=shape[shape.length-1];}
    change({...graph,nodes:graph.nodes.map(n=>n.id===target?{...n,inputs,params}:n)});setConnecting('');
  }
  function remove(){if(!node||node.id==='input')return;change({...graph,nodes:graph.nodes.filter(n=>n.id!==node.id).map(n=>({...n,inputs:n.inputs.filter(i=>i!==node.id)}))});setSelected('input');}
  function fit(){const el=canvas.current;if(!el)return;const minX=Math.min(...graph.nodes.map(n=>n.x)),minY=Math.min(...graph.nodes.map(n=>n.y));const w=Math.max(...graph.nodes.map(n=>n.x+210))-minX,h=Math.max(...graph.nodes.map(n=>n.y+130))-minY;const z=Math.min(1.3,(el.clientWidth-80)/w,(el.clientHeight-80)/h);setCamera({z,x:(el.clientWidth-w*z)/2-minX*z,y:(el.clientHeight-h*z)/2-minY*z});}
  async function load(className?:string){if(!importModel)return;if(!className&&!window.confirm('Replace the current canvas with this model? Your current graph can be recovered with Undo.'))return;setBusy(true);try{const r=await post<{graph?:Graph;classes?:string[]}>('/network-builder/import',{model:importModel,class_name:className});if(r.graph){change(r.graph);setSelected(r.graph.nodes[0]?.id??'');setClasses([]);setNotice('Model imported. Export under a new name to keep the original.');}else setClasses(r.classes??[]);}catch(e){setNotice((e as Error).message);}finally{setBusy(false);}}
  async function exportModel(){setBusy(true);try{const r=await post<{path:string}>('/network-builder/export',{name:graph.name,graph});setNotice(`Exported ${r.path}. The model is now available in Models.`);setModels(m=>[...new Set([...m,graph.name])]);}catch(e){setNotice((e as Error).message);}finally{setBusy(false);}}
  const groups=[...new Set(catalog.map(c=>c.group))];
  return <div className="builder">
    <div className="builder-heading"><div><small>VISUAL ARCHITECTURE / PYTORCH {version}</small><h1>Network Builder</h1></div><div><button className="button small" onClick={()=>setToolbar(!toolbar)}>{toolbar?'Hide':'Show'} toolbar</button> <button className="button small" onClick={()=>setRight(!right)}>{right?'Hide':'Show'} inspector</button></div></div>
    {toolbar&&<div className="builder-toolbar">
      <input aria-label="Network name" value={graph.name} onChange={e=>change({...graph,name:e.target.value})}/>
      <button onClick={fit}>Fit graph</button><button onClick={()=>setCamera({x:0,y:0,z:1})}>Reset view</button>
      <button disabled={!history.length} onClick={()=>{const g=history.at(-1);if(g){setHistory(history.slice(0,-1));setFuture([...future,graph]);setGraph(g);}}}>Undo</button>
      <button disabled={!future.length} onClick={()=>{const g=future.at(-1);if(g){setFuture(future.slice(0,-1));setHistory([...history,graph]);setGraph(g);}}}>Redo</button>
      <button onClick={()=>{if(window.confirm('Clear the graph? The input node will remain.')){change({...graph,nodes:[graph.nodes.find(n=>n.id==='input')??initial().nodes[0]]});setSelected('input');}}}>Clear</button>
      <select aria-label="Existing model" value={importModel} onChange={e=>{setImportModel(e.target.value);setClasses([]);}}><option value="">Import existing model…</option>{models.map(m=><option key={m}>{m}</option>)}</select>
      <button disabled={!importModel||busy} onClick={()=>void load()}>Import</button>
      <button disabled={busy||!!error||!metrics} onClick={()=>void exportModel()}>Export model</button>
      <button onClick={()=>graphFile.current?.click()}>Open graph</button>
      <input type="file" accept=".json" ref={graphFile} hidden onChange={async e=>{const file=e.target.files?.[0];if(!file)return;try{if(file.size>500000)throw new Error('Graph file is too large');const g=JSON.parse(await file.text()) as Graph;if(g.version!==1||!Array.isArray(g.nodes)||!g.nodes.length||g.nodes.some(n=>typeof n.id!=='string'||!Array.isArray(n.inputs)||typeof n.x!=='number'||typeof n.y!=='number'||!n.params))throw new Error('Invalid graph document');await post('/network-builder/analyze',g);change(g);setSelected(g.nodes[0].id);}catch(err){setNotice((err as Error).message);}finally{e.target.value='';}}}/>
      <button onClick={()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(graph,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`${graph.name}.graph.json`;a.click();URL.revokeObjectURL(url);}}>Download graph</button>
    </div>}
    {classes.length>0&&<div className="builder-message">Select class: {classes.map(c=><button key={c} onClick={()=>void load(c)}>{c}</button>)}</div>}
    {notice&&<div className="builder-message" role="status">{notice}<button onClick={()=>setNotice('')}>Dismiss</button></div>}
    <div className={`builder-workspace ${right?'':'inspector-hidden'}`}>
      <aside className="builder-categories"><small>COMPONENTS</small>{groups.map(g=><button className={g===group?'active':''} key={g} onClick={()=>setGroup(g)}>{g}</button>)}<button onClick={()=>add('Input')}>+ Additional input</button></aside>
      <div className="builder-stage">
        <div className="builder-canvas" ref={canvas} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();const type=e.dataTransfer.getData('application/network-node');if(!catalog.some(c=>c.name===type))return;const b=e.currentTarget.getBoundingClientRect();add(type,(e.clientX-b.left-camera.x)/camera.z,(e.clientY-b.top-camera.y)/camera.z);}}
          onWheel={e=>{const b=e.currentTarget.getBoundingClientRect();const x=e.clientX-b.left,y=e.clientY-b.top;setCamera(c=>{const z=Math.max(.15,Math.min(2.5,c.z*(e.deltaY>0?.9:1.1)));return {z,x:x-(x-c.x)*z/c.z,y:y-(y-c.y)*z/c.z};});}}
          onPointerDown={e=>{if(e.target===e.currentTarget){drag.current={x:e.clientX,y:e.clientY,ox:camera.x,oy:camera.y};e.currentTarget.setPointerCapture(e.pointerId);setConnecting('');}}}
          onPointerMove={e=>{const d=drag.current;if(!d)return;if(d.id)setGraph(g=>({...g,nodes:g.nodes.map(n=>n.id===d.id?{...n,x:d.ox+(e.clientX-d.x)/camera.z,y:d.oy+(e.clientY-d.y)/camera.z}:n)}));else setCamera(c=>({...c,x:d.ox+e.clientX-d.x,y:d.oy+e.clientY-d.y}));}}
          onPointerUp={()=>{drag.current=null;}}>
          <div className="builder-world" style={{transform:`translate(${camera.x}px,${camera.y}px) scale(${camera.z})`}}>
            <svg className="builder-wires">{graph.nodes.flatMap(n=>n.inputs.map((id,index)=>{const a=graph.nodes.find(v=>v.id===id);if(!a)return null;const x=a.x+210,y=a.y+58,tx=n.x,ty=n.y+58;return <g key={`${id}-${n.id}`}><path d={`M ${x} ${y} C ${x+85} ${y}, ${tx-85} ${ty}, ${tx} ${ty}`} /><text x={(x+tx)/2} y={(y+ty)/2-8} onClick={()=>change({...graph,nodes:graph.nodes.map(v=>v.id===n.id?{...v,inputs:v.inputs.filter((_,i)=>i!==index)}:v)})}>× {index+1}</text></g>;}))}</svg>
            {graph.nodes.map(n=><div key={n.id} className={`builder-node ${selected===n.id?'selected':''} ${n.type==='Input'?'input-node':''}`} style={{left:n.x,top:n.y}} onClick={()=>{setSelected(n.id);setRight(true);}}>
              <div className="builder-node-title" onPointerDown={e=>{e.stopPropagation();setHistory(h=>[...h,graph]);setFuture([]);drag.current={id:n.id,x:e.clientX,y:e.clientY,ox:n.x,oy:n.y};canvas.current?.setPointerCapture(e.pointerId);setSelected(n.id);}}><strong>{n.type}</strong><small>{n.id==='input'?'LOCKED INPUT':n.id.slice(-6)}</small></div>
              <div className="builder-node-body">{n.type!=='Input'&&<button className={`port in ${connecting?'ready':''}`} title="Connect selected output here" aria-label={`Connect to ${n.type}`} onClick={e=>{e.stopPropagation();connect(n.id);}}>●</button>}<span>{metrics?.nodes[n.id]?.shape.join(' × ')||'Connect / configure'}</span>{n.type!=='Output'&&<button className={`port out ${connecting===n.id?'ready':''}`} title="Start connection" aria-label={`Connect from ${n.type}`} onClick={e=>{e.stopPropagation();setConnecting(n.id);}}>●</button>}</div>
              <small className="builder-node-detail">{Object.entries(n.params).filter(([k])=>['in_features','out_features','in_channels','out_channels','dtype'].includes(k)).map(([k,v])=>`${k}: ${v}`).join(' · ')||'Select to configure'}</small>
            </div>)}
          </div>
          <div className="builder-canvas-help">{connecting?'Click an input port to connect. Click background to cancel.':'Drag nodes · Drag background to pan · Scroll to zoom · Click a wire × to disconnect'} <span>{Math.round(camera.z*100)}%</span></div>
        </div>
        <div className="builder-palette"><div><strong>{group}</strong><input placeholder="Search all components…" aria-label="Search components" value={search} onChange={e=>setSearch(e.target.value)}/></div><div className="builder-items">{catalog.filter(c=>(search?c.name.toLowerCase().includes(search.toLowerCase()):c.group===group)).map(c=><button draggable onDragStart={e=>e.dataTransfer.setData('application/network-node',c.name)} onClick={()=>add(c.name)} title={`Drag onto canvas or click to add. Required: ${c.required.join(', ')||'none'}`} key={c.name}>{c.name}<small>{c.required.length?'Configurable module':'Ready to configure'}</small></button>)}</div></div>
      </div>
      {right&&<aside className="builder-inspector"><h3>Network information</h3>{metrics?<dl>{[['Parameters',metrics.parameters.toLocaleString()],['Weights',bytes(metrics.weight_bytes)],['Buffers',bytes(metrics.buffer_bytes)],['Activations',bytes(metrics.activation_bytes)],['Inference estimate',bytes(metrics.inference_bytes)],['Training estimate (Adam)',bytes(metrics.training_bytes)]].map(([k,v])=><div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>:<p>Complete valid connections to calculate metrics.</p>}<small>Memory assumes float32 weights, input precision as configured, all activations retained and Adam state. Backend workspaces and allocator overhead are excluded. Estimates are not measured peaks.</small>
        {error&&<p className="builder-error" role="alert">{error}</p>}
        {node&&<div className="builder-settings" key={node.id}><h3>{node.type} settings</h3><small>{node.id}</small>{node.type!=='Input'&&<label>Connections (port order){node.inputs.map((id,i)=><button key={id} onClick={()=>editNode({inputs:node.inputs.filter((_,j)=>j!==i)})}>{i+1}: {graph.nodes.find(n=>n.id===id)?.type} ×</button>)}</label>}
          {Object.entries(node.params).map(([key,value])=><label key={key}>{key}<input defaultValue={JSON.stringify(value)} onBlur={e=>{try{const v=JSON.parse(e.target.value);setFieldError('');if(JSON.stringify(v)!==JSON.stringify(value))editNode({params:{...node.params,[key]:v}});}catch{setFieldError(`${key}: enter JSON (strings need quotes, e.g. "float32")`);}}}/></label>)}
          <small>Values use JSON: numbers, true/false, null, arrays, or quoted strings. Input shape includes batch size.</small>{fieldError&&<p className="builder-error">{fieldError}</p>}
          <button disabled={node.id==='input'} onClick={remove}>Delete node</button>
        </div>}
        <p className="builder-support">Module availability follows installed PyTorch. Modules needing other modules, callable arguments, custom control flow, or tuple unpacking may require a custom adapter. Validation reports unsupported configurations.</p>
      </aside>}
    </div>
  </div>;
}
