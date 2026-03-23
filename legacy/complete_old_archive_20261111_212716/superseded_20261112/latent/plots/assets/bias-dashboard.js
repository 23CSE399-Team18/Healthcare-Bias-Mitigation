'use strict';
(() => {
  const data = JSON.parse(document.getElementById('payload').textContent);
  const el = id => document.getElementById(id);
  const fmt = x => x == null ? 'NA' : typeof x === 'number' ? Number.isInteger(x) ? x.toLocaleString() : x.toPrecision(4) : String(x);
  const beforeColor = '#77ade0', afterColor = '#83b657';
  let current, generation = 0;
  function node(tag, text, className) { const n=document.createElement(tag); if(text != null)n.textContent=text; if(className)n.className=className; return n; }
  function options(select, values, selected) {select.replaceChildren(); for(const [value,label] of values){const o=node('option',label);o.value=value;select.append(o);}select.value=values.some(v=>v[0]===selected)?selected:values[0]?.[0] || '';}
  function status(id, label, value) {
    const box=node('div',null,'status '+(value.label==='Detected by rule'?'detected':value.label==='Not detected by rule'?'clear':''));
    box.append(node('strong',label),node('span',value.label),node('span','Value: '+fmt(value.value)),node('span',value.rule));
    el(id).replaceChildren(box);
  }
  function tablePanel(panel) {
    const wrap=node('div',null,'table-wrap'), table=node('table'), head=node('thead'), body=node('tbody');
    const columns=[...new Set(panel.table.flatMap(r=>Object.keys(r)))];const tr=node('tr');columns.forEach(c=>tr.append(node('th',c.replaceAll('_',' '))));head.append(tr);
    for(const row of panel.table){const r=node('tr');for(const c of columns){const value=row[c];r.append(node('td',value && typeof value==='object'?JSON.stringify(value):fmt(value),value==null?'na':''));}body.append(r);}
    table.append(head,body);wrap.append(table);return wrap;
  }
  function distributionPanels(kind, variable) {
    const v=current.variables[variable]; if(!v)return [];
    const labels=[current.before_label,current.after_label];
    if(kind==='hist' || kind==='density') {
      const centers=v.edges.slice(1).map((x,i)=>(x+v.edges[i])/2), widths=v.edges.slice(1).map((x,i)=>x-v.edges[i]);
      return [{title:variable+' — '+(kind==='density'?'binned probability density':'full-range histogram'),data:[['before',beforeColor],['after',afterColor]].map(([key,color],i)=>({type:'bar',x:centers,y:v[key+'_counts'].map((count,j)=>kind==='density'?count/(Math.max(v[key+'_n'],1)*widths[j]):count),width:widths.map(w=>w*.96),name:labels[i]+' (n='+v[key+'_n'].toLocaleString()+')',marker:{color},opacity:.60,customdata:v.edges.slice(0,-1).map((a,j)=>[a,v.edges[j+1]]),hovertemplate:'Bin [%{customdata[0]}, %{customdata[1]}]<br>%{y}<extra>%{fullData.name}</extra>'})),layout:{barmode:'overlay',xaxis:{title:variable},yaxis:{title:kind==='density'?'Probability density (counts / n / bin width)':'Rows'},height:420}}];
    }
    if(kind==='box') return [{title:variable+' — quartiles and full range',data:[['before',beforeColor],['after',afterColor]].filter(([key])=>v[key+'_quantiles']).map(([key,color])=>{const q=v[key+'_quantiles'];return {type:'box',name:labels[key==='before'?0:1],q1:[q[25]],median:[q[50]],q3:[q[75]],lowerfence:[q[0]],upperfence:[q[100]],boxpoints:false,marker:{color},line:{color}};}),layout:{yaxis:{title:variable},height:420},note:'Whiskers show the full minimum–maximum range, not Tukey outlier fences. Quartiles are computed from every finite value.'}];
    if(kind==='qq') {
      if(!v.before_quantiles || !v.after_quantiles)return [{title:'Q-Q comparison',table:[{status:'Unavailable: one series has no finite observations'}]}];
      const all=v.before_quantiles.concat(v.after_quantiles),lo=Math.min(...all),hi=Math.max(...all);
      return [{title:variable+' — before/after quantile comparison',data:[{type:'scatter',mode:'lines',x:[lo,hi],y:[lo,hi],name:'Equal quantiles',line:{dash:'dash',color:'#888'}},{type:'scatter',mode:'markers',x:v.before_quantiles,y:v.after_quantiles,name:'0–100% quantiles',marker:{color:'#536ab1',size:6},customdata:Array.from({length:101},(_,i)=>i),hovertemplate:'Quantile %{customdata}%<br>Before %{x}<br>After %{y}<extra></extra>'}],layout:{xaxis:{title:current.before_label},yaxis:{title:current.after_label},height:440},note:'This compares the two empirical distributions. It is not a normality test.'}];
    }
    return [];
  }
  async function render() {
    const revision=++generation;current=data.states[el('split').value][el('method').value];
    options(el('variable'),Object.keys(current.variables).map(k=>[k,k]),el('variable').value);
    const hasVariables=Object.keys(current.variables).length>0;
    for(const option of el('view').options)option.disabled=option.value!=='summary'&&!hasVariables;
    if(!hasVariables)el('view').value='summary';el('variable').disabled=!hasVariables;
    status('before',current.before_label,current.before);status('after',current.after_label,current.after);
    el('info').replaceChildren();
    for(const [label,value] of [['Partition',el('split').value],['Rows',current.rows],['Input features',current.input_features],[data.stage==='preprocessing'?'Numeric inputs':'Latent dimensions',current.numeric_features]]){if(value!=null){const row=node('div',null,'info-row');row.append(node('span',label),node('strong',fmt(value)));el('info').append(row);}}
    el('explanation').replaceChildren(node('h2',data.title),node('p',data.protocol),node('p',current.note,'rule'));
    if(current.thresholds)el('explanation').append(node('p','Frozen thresholds: ERM '+fmt(current.thresholds.ERM)+'; comparison '+fmt(current.thresholds.comparison)+'.','small'));
    for(const chart of el('plots').querySelectorAll('.chart'))Plotly.purge(chart);el('plots').replaceChildren();
    const panels=el('view').value==='summary'?current.panels:distributionPanels(el('view').value,el('variable').value);
    if(!panels.length)el('plots').append(node('div','This metric was not computed for this branch. It is unavailable, not zero.','empty'));
    for(const panel of panels) {
      if(revision!==generation)return;
      const card=node('section',null,'plot-card');card.append(node('h2',panel.title));el('plots').append(card);
      if(panel.table){if(panel.table.length)card.append(tablePanel(panel));else card.append(node('p','No supported observations for this comparison.','empty'));}
      else {
        const visible=panel.data.some(t=>(t.y||t.z||t.q1||[]).flat().some(v=>v!=null&&Number.isFinite(typeof v==='number'?v:Number(v)))) || panel.data.some(t=>t.orientation==='h'&&(t.x||[]).some(v=>v!=null));
        if(!visible){card.append(node('p','No defined values for this comparison. Missing values are not plotted as zero.','empty'));continue;}
        const chart=node('div',null,'chart');card.append(chart);
        const layout={paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{family:'Segoe UI, sans-serif',color:'#263b55',size:12},margin:{l:90,r:30,t:30,b:100},legend:{orientation:'h',y:1.13},...panel.layout};
        layout.xaxis={...layout.xaxis,automargin:true};layout.yaxis={...layout.yaxis,automargin:true};
        for(const axis of [layout.xaxis,layout.yaxis])if(typeof axis.title==='string')axis.title={text:axis.title};
        await Plotly.newPlot(chart,panel.data,layout,{responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',filename:data.stage+'_'+data.dataset.replaceAll('/','_')+'_'+data.kind,scale:2}});
      }
      if(panel.note)card.append(node('p',panel.note,'small'));
    }
    window.dashboardReady=true;
  }
  function chooseSplit(){options(el('method'),Object.keys(data.states[el('split').value]).map(k=>[k,data.methods[k]||k]),el('method').value||data.default_method);el('method').disabled=el('method').options.length===1;return render();}
  function safely(fn){return ()=>Promise.resolve().then(fn).catch(error=>{el('plots').replaceChildren(node('div','The dashboard could not render: '+error.message,'error'));window.dashboardError=error.message;});}
  el('title').textContent=data.title+' — '+data.dataset.replaceAll('_',' ');el('subtitle').textContent='Individual '+data.stage+' dashboard | verified current data';
  options(el('split'),Object.keys(data.states).map(k=>[k,k]),data.default_split);
  for(const id of ['method','view','variable'])el(id).addEventListener('change',safely(render));el('split').addEventListener('change',safely(chooseSplit));
  el('download').addEventListener('click',()=>{const blob=new Blob([JSON.stringify({dataset:data.dataset,bias:data.kind,split:el('split').value,method:el('method').value,...current},null,2)],{type:'application/json'});const a=node('a');a.href=URL.createObjectURL(blob);a.download=data.stage+'_'+data.kind+'_plot_data.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);});
  safely(chooseSplit)();
})();
