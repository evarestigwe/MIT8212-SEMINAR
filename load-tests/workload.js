import http from "k6/http";
import { check } from "k6";

const rate = Number(__ENV.RATE || 1);
const duration = __ENV.DURATION || "1m";
const target =
  __ENV.TARGET ||
  "http://localhost:8080/work";

export const options = {
  scenarios: {
    application_workload: {
      executor: "constant-arrival-rate",
      rate,
      timeUnit: "1s",
      duration,
      preAllocatedVUs: 20,
      maxVUs: 100,
    },
  },
};

export default function () {
  const response = http.get(target, {
    timeout: "30s",
  });

  check(response, {
    "status is 200": (r) =>
      r.status === 200,
  });
}