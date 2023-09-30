#!/usr/bin/env python

import logging
import pynvml as nv
from rich.pretty import install


install()
logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)

def get_reason(val):
    throttle = {
        1: 'gpu idle',
        2: 'applications clocks setting',
        4: 'sw power cap',
        8: 'hw slowdown',
        16: 'sync boost',
        32: 'sw thermal slowdown',
        64: 'hw thermal slowdown',
        128: 'hw power brake slowdown',
        256: 'display clock setting',
    }
    reason = ', '.join([throttle[i] for i in throttle if i & val])
    return reason if len(reason) > 0 else 'none'

def main():
    nv.nvmlInit()
    log.info(f"version cuda={nv.nvmlSystemGetCudaDriverVersion()} driver={nv.nvmlSystemGetDriverVersion()}")
    for i in range(nv.nvmlDeviceGetCount()):
        dev = nv.nvmlDeviceGetHandleByIndex(i)
        log.info(f"device#{i}: {nv.nvmlDeviceGetName(dev)}")
        log.info(f" version vbios={nv.nvmlDeviceGetVbiosVersion(dev)} rom={nv.nvmlDeviceGetInforomImageVersion(dev)}")
        log.info(f" cuda capabilities: {nv.nvmlDeviceGetCudaComputeCapability(dev)}")
        log.info(f" pci link={nv.nvmlDeviceGetCurrPcieLinkGeneration(dev)} width={nv.nvmlDeviceGetCurrPcieLinkWidth(dev)} busid={nv.nvmlDeviceGetPciInfo(dev).busId} deviceid={nv.nvmlDeviceGetPciInfo(dev).pciDeviceId}")
        log.info(f" memory total={round(nv.nvmlDeviceGetMemoryInfo(dev).total/1024/1024, 2)}Mb free={round(nv.nvmlDeviceGetMemoryInfo(dev).free/1024/1024,2)}Mb used={round(nv.nvmlDeviceGetMemoryInfo(dev).used/1024/1024,2)}Mb")
        log.info(f" clock graphics={nv.nvmlDeviceGetClockInfo(dev, 0)}Mhz sm={nv.nvmlDeviceGetClockInfo(dev, 1)}Mhz memory={nv.nvmlDeviceGetClockInfo(dev, 2)}Mhz")
        log.info(f" utilization gpu={nv.nvmlDeviceGetUtilizationRates(dev).gpu}% memory={nv.nvmlDeviceGetUtilizationRates(dev).memory}% temp={nv.nvmlDeviceGetTemperature(dev, 0)}C fan={nv.nvmlDeviceGetFanSpeed(dev)}%")
        log.info(f" power usage={round(nv.nvmlDeviceGetPowerUsage(dev)/1000, 2)}w limit={round(nv.nvmlDeviceGetEnforcedPowerLimit(dev)/1000, 2)}w energy={round(nv.nvmlDeviceGetTotalEnergyConsumption(dev)/1000000000, 2)}Mj")
        log.info(f" throttle={get_reason(nv.nvmlDeviceGetCurrentClocksThrottleReasons(dev))} state power={nv.nvmlDeviceGetPowerState(dev)} performance={nv.nvmlDeviceGetPerformanceState(dev)}")
    nv.nvmlShutdown()


if __name__ == "__main__":
    main()
