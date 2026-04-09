let selectedFile = null;
let chart = null;
let lastPrediction = null;

// FILE UPLOAD
function triggerUpload() {
  document.getElementById("fileInput").click();
}

document.getElementById("fileInput").addEventListener("change", (e) => {
  selectedFile = e.target.files[0];
  if (selectedFile) previewImage(selectedFile);
});

// PREVIEW
function previewImage(file) {
  const reader = new FileReader();

  reader.onload = () => {
    document.getElementById("preview").src = reader.result;
    resetUI();
  };

  reader.readAsDataURL(file);
}

// RESET
function resetUI() {
  document.getElementById("result").innerText = "";
  document.getElementById("confidenceText").innerText = "";
  document.getElementById("fill").style.width = "0%";
  document.getElementById("heatmap").src = "";
  document.getElementById("insights").innerHTML = "";

  document.getElementById("downloadBtn").disabled = true;
  lastPrediction = null;
}

// ANALYZE
async function analyzeImage() {
  if (!selectedFile) return alert("Upload image first");

  document.getElementById("loader").style.display = "block";

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const res = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    if (!res.ok) throw new Error("Prediction failed");

    const data = await res.json();

    document.getElementById("result").innerText = data.label;

    const confidence = data.confidence * 100;
    animateProgress(confidence);

    document.getElementById("confidenceText").innerText =
      "Confidence: " + confidence.toFixed(2) + "%";

    drawGraph(data.tumor_probability, data.no_tumor_probability);

    // 🔥 HEATMAP BACK
    if (data.heatmap) {
      document.getElementById("heatmap").src =
        "data:image/jpeg;base64," + data.heatmap;
    }

    showInsights(data.label);

    lastPrediction = data;
    document.getElementById("downloadBtn").disabled = false;

  } catch (err) {
    alert("Error: " + err.message);
  }

  document.getElementById("loader").style.display = "none";
}

// 🔥 PROGRESS FUNCTION (CORRECT PLACE)
function animateProgress(target) {
  let progress = 0;
  const fill = document.getElementById("fill");

  let interval = setInterval(() => {
    progress += 2;
    fill.style.width = progress + "%";

    if (progress >= target) clearInterval(interval);
  }, 20);
}

// GRAPH
function drawGraph(tumor, noTumor) {
  const ctx = document.getElementById("chart").getContext("2d");

  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Tumor", "No Tumor"],
      datasets: [{
        data: [tumor, noTumor],
        backgroundColor: ["#ff003c", "#00ff9c"],
        borderRadius: 10
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, max: 1 } }
    }
  });
}

// INSIGHTS
function showInsights(label) {
  const insights = document.getElementById("insights");

  if (label === "Tumor Detected") {
    insights.innerHTML = `
      <b>Possible abnormal mass detected</b><br><br>
      - Immediate consultation recommended<br>
      - Further diagnostic evaluation required
    `;
  } else {
    insights.innerHTML = `
      <b>No tumor patterns detected</b><br><br>
      - Brain appears normal<br>
      - Routine monitoring suggested
    `;
  }
}

// PDF DOWNLOAD
async function downloadReport() {
  if (!lastPrediction) return alert("Analyze first");

  const patientName = document.getElementById("patientName").value;
  const doctorNotes = document.getElementById("doctorNotes").value;
  print("🔥 PDF FUNCTION UPDATED")
  

  const res = await fetch("/download-report", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      patientName,
      doctorNotes,
      prediction: lastPrediction
    })
  });

  const blob = await res.blob();

  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "Brain_Tumor_Report.pdf";
  a.click();
}

// THEME
function toggleTheme() {
  document.body.classList.toggle("dark");
}