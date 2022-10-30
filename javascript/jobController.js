/**
 * Controller of the job in progress for the UI
 */
window.JobController = (function () {
  let _jobInProgress = false;
  let eventListeners = [];

  const propagateEvent = (eventName, jobName) => {
    eventListeners.forEach((eventListener) => {
      if (eventListener.eventName === eventName) {
        eventListener.fn(jobName);
      }
    });
  };

  return {
    /**
     * Registers an event listener for the selected event. 
     * @param {string} eventName The name of the event you are listening, it supports "jobStart" and "jobEnd"
     * @param {(jobName: string) => void} fn Function to be called when the event is fired. Receives the name of the job that triggered the event.
     * @returns The registered function, usefull for removing the event listener.
     */
    addEventListener: function (eventName, fn) {
      eventListeners.push({ eventName, fn });
      return fn;
    },

    /**
     * Removes a subscription.
     * @param {function} fn The lsitener to remove, must be the same reference as the registered one to be deleted.
     */
    removeEventListener: function (fn) {
      eventListeners = eventListeners.filter((e) => e.fn !== fn);
    },

    /**
     * Sets the job.
     * @param {string} jobName Name of the job, it will be issued to any registered listener.
     */
    startJob: function (jobName) {
      if (_jobInProgress) {
        throw new Error(`Tried to start a job (${jobName}) while there is one still active (${_jobInProgress})`);
      }

      _jobInProgress = jobName;
      propagateEvent("jobStart", jobName);
    },

    /**
     * End the job.
     */
    endJob: function () {
      if (_jobInProgress) {
        const oldJobName = _jobInProgress;
        _jobInProgress = false;
        propagateEvent("jobEnd", oldJobName);
      }
    },

    /**
     * Return the current job's name, or false if no job is running
     * @returns The current job's name
     */
    getJob: function () {
      return _jobInProgress;
    },
  };
})();
