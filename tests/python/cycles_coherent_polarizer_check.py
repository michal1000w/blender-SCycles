#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Non-render evaluation of declared coherent polarizer batch gates."""
import argparse,json,hashlib
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--partial',action='store_true');a=p.parse_args();plan=json.loads(a.plan.read_text());cases={};passed=True;pending=[]
    state=json.loads(Path(plan['state']).read_text()).get('jobs',{}) if a.partial and Path(plan['state']).exists() else {}
    for job in plan['jobs']:
        args=job['arguments'];worker=Path(args[args.index('--report')+1]);raw=Path(args[args.index('--render')+1]);label=job['case'];gate=job['gate']
        if a.partial and state.get(label,{}).get('status') not in ['completed','error']:
            pending.append(label);continue
        if not worker.exists():entry={'passed':False,'error':'Missing worker report'}
        else:
            result=json.loads(worker.read_text());metrics=result.get('metrics',{});ok=result.get('worker_status')=='completed' and metrics.get('all_finite',False)
            ok &= metrics.get('absolute_mean_error',float('inf'))<=gate['absolute_mean_error']
            ok &= metrics.get('absolute_rmse',float('inf'))<=gate['absolute_rmse']
            if gate.get('absolute_render_mean_max') is not None:ok &= abs(metrics.get('render_mean_radiance',float('inf')))<=gate['absolute_render_mean_max']
            phase=result.get('phase_check',{});phase_gate=job.get('phase_gate')
            if phase_gate:
                ok &= all(phase.get(key,float('inf'))<=limit for key,limit in phase_gate.items())
            entry={'passed':bool(ok),'gate':gate,'metrics':metrics,'phase_gate':phase_gate,'phase_check':phase,'worker_report':str(worker),'worker_sha256':digest(worker),'raw_exr':str(raw),'raw_sha256':digest(raw) if raw.exists() else None,'reference_sha256':job['reference_sha256'],'scene_sha256':job['scene_sha256']}
        cases[label]=entry;passed &= entry['passed']
    report={'scope':'Non-render comparison against prospective physical gate mappings; actual pixels unchanged','plan_sha256':digest(a.plan),'passed':bool(passed),'partial':a.partial,'complete':not pending and len(cases)==len(plan['jobs']),'expected_cases':len(plan['jobs']),'pending':pending,'cases':cases}
    a.report.write_text(json.dumps(report,indent=2)+'\n');print('PASS' if passed else 'FAIL',len(cases),'checked cases;',len(pending),'pending')
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
