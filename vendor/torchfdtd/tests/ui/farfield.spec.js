import {test,expect} from '@playwright/test';
import {readFile} from 'node:fs/promises';
const names=['x_min','x_max','y_min','y_max','z_min','z_max'];
const metadata={job_id:'sample',name:'Stored dipole',isolated_pml:true,background_index:1,interior_bounds_um:[[-2,2],[-2,2],[-2,2]],note:'Six recorded faces.',monitors:names.map((id,i)=>({id,name:id,normal:id[0],position_um:i%2?1:-1,shape:[2,2,1],points:4,components:['Ex','Ey','Ez','Hx','Hy','Hz'],frequency_thz:[200],dtype:'complex64'}))};
const output={request:{version:1},request_digest:'a'.repeat(64),job_id:'sample',reference:null,field_kind:'total',frequency_hz:200e12,frequency_thz:200,refractive_index:1,phase_origin_um:[0,0,0],bounds_um:[[-1,1],[-1,1],[-1,1]],theta_deg:[0,90,180],phi_deg:[0,180],directions:[[0,0,1],[0,0,1],[1,0,0],[-1,0,0],[0,0,-1],[0,0,-1]],electric_real:Array.from({length:6},()=>[1,2,3]),electric_imag:Array.from({length:6},()=>[.1,.2,.3]),intensity:[[0,0],[2,4],[0,0]],relative_intensity:[[0,0],[.5,1],[0,0]],zero_pattern:false,intensity_units:'reduced E*H * s^2 * m^2 / sr',amplitude_units:'reduced E * s * m',report:{points:24,retained_bytes:1152},note:'Continuum projection of stored fields.'};
async function open(page){
 const project=await (await page.request.get('/api/examples/3d')).json();
 await page.route('**/api/jobs',route=>route.fulfill({json:[{id:'sample',name:'Stored dipole',status:'completed',flux_monitors:[]},{id:'incident',name:'Matched incident',status:'completed',flux_monitors:[]}]}));
 await page.route('**/api/jobs/sample',route=>route.fulfill({json:{id:'sample',status:'completed',progress:{step:2,total:2,frame:null},project,monitors:[],flux_monitors:[],summary:{seconds:1,backend:'cpu',mcells_per_second:1,steps:2,setup_seconds:0,warnings:[]}}}));
 await page.route('**/api/jobs/sample/fields',route=>route.fulfill({json:{frames:[],frame_steps:[],epsilon:[[1]],summary:{}}}));
 await page.route('**/api/jobs/sample/farfield-monitors',route=>route.fulfill({json:metadata}));
 await page.addInitScript(p=>{localStorage.setItem('torchfdtd.project.v1',JSON.stringify(p));localStorage.setItem('torchfdtd.activeJob','sample');},project);
 await page.goto('/');await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS');
 await page.locator('[data-action="flux-results"]').first().click();await page.getByRole('button',{name:'Closed-box far field',exact:true}).click();
 const dialog=page.locator('.farfield-dialog');await expect(dialog).toBeVisible();return dialog;
}
async function downloadText(download){return readFile(await download.path(),'utf8');}

test('closed-box controls send exact face and angle metadata and export displayed complex results',async({page})=>{
 let request;await page.route('**/api/jobs/sample/farfield',route=>{request=route.request().postDataJSON();return route.fulfill({json:output});});
 const d=await open(page);
 await expect(d.getByLabel('X minimum face')).toHaveValue('x_min');await expect(d.getByLabel('Z upper bound (µm)')).toHaveValue('1');
 await d.getByRole('button',{name:'Calculate far field'}).click();await expect(d.getByRole('status')).toContainText('Confirm');
 await d.getByLabel('Confirm homogeneous closed surface').check();await d.getByLabel('Theta samples',{exact:true}).fill('3');await d.getByLabel('Phi samples',{exact:true}).fill('2');
 await d.getByRole('button',{name:'Calculate far field'}).click();await expect(d.getByRole('heading',{name:'Angular radiation pattern'})).toBeVisible();
 await expect(d.getByRole('status')).toContainText('Total fields');expect(request.faces).toEqual(Object.fromEntries(names.map(n=>[n,n])));expect(request.bounds_um).toEqual([[-1,1],[-1,1],[-1,1]]);expect(request.theta_deg).toEqual({start:0,stop:180,count:3});expect(request.phi_deg).toEqual({start:0,stop:360,count:2});expect(request.subtract_incident).toBe(false);
 await expect(d).toContainText('not calibrated W/sr or an efficiency');await d.getByLabel('Far-field display scale').selectOption('raw');await expect(d.locator('[data-range]')).toContainText('4.000000e+0');
 const csv=page.waitForEvent('download');await d.getByRole('button',{name:'Export direction CSV'}).click();const text=await downloadText(await csv);expect(text).toContain('intensity_reduced_EH_s2_m2_per_sr');expect(text.split('\n')[4]).toBe('90,180,-1,0,0,4,1,1,2,3,0.1,0.2,0.3');
 const json=page.waitForEvent('download');await d.getByRole('button',{name:'Export result JSON'}).click();expect(JSON.parse(await downloadText(await json))).toEqual(output);
 const setup=page.waitForEvent('download');await d.getByRole('button',{name:'Export setup JSON'}).click();expect(JSON.parse(await downloadText(await setup))).toEqual({job_id:'sample',request});
});

test('coherent reference selection is explicit and a zero pattern has finite plots and CSV',async({page})=>{
 let request,calls=0;const zero={...output,reference:'incident',field_kind:'scattered',zero_pattern:true,intensity:[[0,0],[0,0],[0,0]],relative_intensity:[[0,0],[0,0],[0,0]]};
 await page.route('**/api/jobs/sample/farfield',route=>{calls++;request=route.request().postDataJSON();return route.fulfill({json:zero});});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));const d=await open(page);await d.getByLabel('Confirm homogeneous closed surface').check();await d.getByLabel('Far-field reference').selectOption('incident');
 await d.getByRole('button',{name:'Calculate far field'}).click();await expect(d.getByRole('status')).toContainText('enable complex-field subtraction');expect(calls).toBe(0);
 await d.getByLabel('Subtract matched incident fields').check();await d.getByRole('button',{name:'Calculate far field'}).click();await expect(d.getByRole('status')).toContainText('Scattered fields');await expect(d.getByRole('status')).toContainText('Zero pattern');expect(request.reference).toBe('incident');expect(request.subtract_incident).toBe(true);await expect(d.locator('[data-range]')).toContainText('0.000000e+0');
 await d.getByLabel('Theta cut at phi').selectOption('1');await d.getByLabel('Phi cut at theta').selectOption('1');expect(errors).toEqual([]);
 const csv=page.waitForEvent('download');await d.getByRole('button',{name:'Export direction CSV'}).click();expect(await downloadText(await csv)).not.toMatch(/NaN|Infinity/);
});

test('edits and closing discard pending projection without starting a simulation or claiming cancellation',async({page})=>{
 let release,started,posts=0;const start=new Promise(resolve=>started=resolve),held=new Promise(resolve=>release=resolve);
 await page.route('**/api/jobs/sample/farfield',async route=>{posts++;started();await held;await route.fulfill({json:output});});
 const d=await open(page);await d.getByLabel('Confirm homogeneous closed surface').check();await d.getByRole('button',{name:'Calculate far field'}).click();await start;await d.getByLabel('Theta samples',{exact:true}).fill('5');release();await expect(d.getByRole('status')).toContainText('Setup changed');await expect(d.locator('[data-results]')).toBeEmpty();expect(posts).toBe(1);
 await d.getByLabel('Phi start (degrees)',{exact:true}).fill('-1');await d.getByRole('button',{name:'Calculate far field'}).click();await expect(d.getByRole('status')).toContainText('Phi must increase within 0–360');expect(posts).toBe(1);
 await d.getByLabel('Phi start (degrees)',{exact:true}).fill('0');await d.getByLabel('Phi stop (degrees, excluded)',{exact:true}).fill('361');await d.getByRole('button',{name:'Calculate far field'}).click();await expect(d.getByRole('status')).toContainText('Phi must increase within 0–360');expect(posts).toBe(1);
 await d.getByLabel('Phi stop (degrees, excluded)',{exact:true}).fill('360');
 await d.getByLabel('Theta samples',{exact:true}).fill('8192');await d.getByRole('button',{name:'Calculate far field'}).click();await expect(d.getByRole('status')).toContainText('at most 8192');expect(posts).toBe(1);
 await page.unroute('**/api/jobs/sample/farfield');let releaseClose,startedClose;const startClose=new Promise(resolve=>startedClose=resolve),heldClose=new Promise(resolve=>releaseClose=resolve);
 await page.route('**/api/jobs/sample/farfield',async route=>{posts++;startedClose();await heldClose;await route.fulfill({json:output});});
 await d.getByLabel('Theta samples',{exact:true}).fill('5');await d.getByRole('button',{name:'Calculate far field'}).click();await startClose;
 await expect(d).toContainText('does not cancel server calculation');await d.getByRole('button',{name:'Close',exact:true}).click();const completed=page.waitForResponse(r=>r.url().endsWith('/farfield'));releaseClose();await completed;await expect(d).toHaveCount(0);expect(posts).toBe(2);
});
