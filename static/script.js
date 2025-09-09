//XỬ LÝ TÍNH NĂNG TỰ ĐỘNG ĐẾM KIM CƯƠNG VỚI KHAY VÀ ACTIVE LEARNING
// Tạo một biến toàn cục để lưu trữ vùng khay
window.trayRegion = null;
window.currentPredictions = []; // Store current predictions for feedback
window.userCorrections = {
    false_positives: [],
    missed_objects: []
};

// Hàm này sẽ được gọi khi người dùng xác nhận vùng khay
// region_x, region_y, region_w, region_h là tọa độ và kích thước của vùng khay
// tray_width, tray_length là kích thước của khay
function startAutoCountWithTray(region_x, region_y, region_w, region_h, tray_width, tray_length) {
    window.trayRegion = { region_x, region_y, region_w, region_h, tray_width, tray_length };
    autoCountDiamonds();
}
// Hàm này sẽ tự động đếm kim cương trong vùng đã chọn
// Nó sẽ gửi yêu cầu đến server để xử lý ảnh và trả về kết quả
async function autoCountDiamonds() {
    setUIEnabled(false);
    try {
        await fetch("/reset", { method: "POST" });
        const canvas = document.getElementById("canvasUpload");
        const partSize = parseInt(document.getElementById("partSizeSelect").value);
        const stride = partSize / 2;

        let region_x = 0, region_y = 0, region_w = canvas.width, region_h = canvas.height;
        if (window.trayRegion) {
            region_x = window.trayRegion.region_x;
            region_y = window.trayRegion.region_y;
            region_w = window.trayRegion.region_w;
            region_h = window.trayRegion.region_h;
            trayLength = window.trayRegion.tray_length;
            trayWidth = window.trayRegion.tray_width;
        }

        const fontSize = parseInt(document.getElementById("fontSize").value);
        const minSize = parseFloat(document.getElementById("minSize").value);
        const circleThreshold = parseFloat(document.getElementById("circleThreshold").value);


        let totalDiamonds = 0;

        // Thêm class scanning cho animation
        const box = document.getElementById("resizableBox");
        box.classList.add("scanning");

        // Lấy scale giữa canvas hiển thị (CSS) và canvas nội bộ (width 640)
        const scale = canvas.clientWidth / canvas.width;

        let promises = [];
        for (let y = 0; y <= canvas.height - partSize; y += stride) {
            for (let x = 0; x <= canvas.width - partSize; x += stride) {

                // Hiển thị box trên giao diện theo tỷ lệ CSS
                box.style.left = (x * scale) + "px";
                box.style.top = (y * scale) + "px";
                box.style.width = (partSize * scale) + "px";
                box.style.height = (partSize * scale) + "px";

                await new Promise(resolve => setTimeout(resolve, 25)); // Tăng tốc animation từ 100ms xuống 25ms

                // Cắt toàn ảnh (vì YOLO sẽ xử lý qua tọa độ vùng)
                const promise = (async () => {
                    const canvasBlob = await new Promise(resolve =>
                        canvas.toBlob(resolve, "image/png")
                    );
                    const formData = new FormData();
                    formData.append("image", canvasBlob, `part_${x}_${y}.png`);
                    formData.append("tray_length", trayLength);
                    formData.append("tray_width", trayWidth);
                    formData.append("min_size", minSize);
                    formData.append("font_size", fontSize);
                    formData.append("region_x", x);
                    formData.append("region_y", y);
                    formData.append("region_w", partSize);
                    formData.append("region_h", partSize);
                    formData.append("circle_threshold", circleThreshold);



                    try {
                        const res = await fetch("/predict?mode=box", {
                            method: "POST",
                            body: formData,
                        });

                    } catch (error) {
                        console.error(`Lỗi khi xử lý vùng (${x}, ${y})`, error);
                        return null;
                    }
                })();
                promises.push(promise);
            }
        }
        await Promise.all(promises);
        const formData = new FormData();
        formData.append("region_x", region_x);
        formData.append("region_y", region_y);
        formData.append("region_w", region_w);
        formData.append("region_h", region_h);
        formData.append("tray_width", trayWidth);
        formData.append("tray_length", trayLength);
        formData.append("min_size", minSize);
        formData.append("circle_threshold", circleThreshold);


        // Lấy file ảnh gốc từ input
        const fileInput = document.getElementById("fileInput");
        const originalFile = fileInput.files[0];
        if (originalFile) {
            formData.append("original_image", originalFile, originalFile.name);
        }

        // Gửi ảnh đã xử lý lên server để lấy kết quả cuối cùng
        const res = await fetch("/final_result", {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        if (data.annotated_image) {
            const img = new Image();
            img.onload = () => originalAnnotatedImageOnload(img);
            img.src = "data:image/jpeg;base64," + data.annotated_image;
        }
        
        document.getElementById("totalCountDisplay").innerText = `Tổng số: ${data.total_diamonds}💎`;

        // Hiển thị bảng chi tiết
        const tbody = document.getElementById("resultsTableBody");
        tbody.innerHTML = "";
        if (Array.isArray(data.details)) {
            data.details.forEach((item) => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
      <td style="padding-left: 20px;">${item.index}</td>
      <td style="padding-left: 20px;">${item.width?.toFixed(2) ?? ""}</td>
      <td style="padding-left: 20px;">${item.height?.toFixed(2) ?? ""}</td>
      <td style="padding-left: 20px;">${item.shape ?? ""}</td>
      <td style="padding-left: 20px;">${item.area?.toFixed(2) ?? ""}</td>
      <td style="padding-left: 20px;">${item.diameter?.toFixed(2) ?? ""}</td>
    `;
                tbody.appendChild(tr);
            });
        }
    } finally {
        // Remove scanning animation khi hoàn thành
        const box = document.getElementById("resizableBox");
        box.classList.remove("scanning");
        
        setUIEnabled(true);
    }
}


  document.getElementById("saveExcelBtn").addEventListener("click", function () {
    // Lấy bảng
    var table = document.querySelector("table");

    // Chuyển bảng HTML thành sheet
    var wb = XLSX.utils.table_to_book(table, { sheet: "Kết quả" });

    // Tạo tên file kèm ngày giờ
    var now = new Date();
    var fileName = "ket_qua_" 
      + now.getFullYear() + "-"
      + String(now.getMonth() + 1).padStart(2, '0') + "-"
      + String(now.getDate()).padStart(2, '0') + "_"
      + String(now.getHours()).padStart(2, '0') + "-"
      + String(now.getMinutes()).padStart(2, '0') + "-"
      + String(now.getSeconds()).padStart(2, '0')
      + ".xlsx";

    // Xuất file
    XLSX.writeFile(wb, fileName);
  });




// Xử lý sự kiện khi người dùng nhấn nút "Tự động cắt khay"
document.getElementById('autoCutBtn').addEventListener('click', function () {
    document.getElementById('cutOptionsGroup').classList.add('show');
    document.getElementById('confirmTrayBtn').style.display = '';
});


// Xử lý sự kiện khi người dùng xác nhận vùng khay
document.getElementById("confirmTrayBtn").addEventListener("click", function () {
    document.getElementById('cutOptionsGroup').classList.remove('show');
    const scaleFactor = TARGET_SIZE / 320;
    const region_x = Math.round(box.offsetLeft * scaleFactor);
    const region_y = Math.round(box.offsetTop * scaleFactor);
    const region_w = Math.round(box.offsetWidth * scaleFactor);
    const region_h = Math.round(box.offsetHeight * scaleFactor);

    const tray_width = parseFloat(document.getElementById("trayWidth").value);
    const tray_length = parseFloat(document.getElementById("trayLength").value);
    const minSize = parseFloat(document.getElementById("minSize").value);

    // Lưu vào biến toàn cục để autoCountDiamonds dùng
    window.trayRegion = { region_x, region_y, region_w, region_h, tray_width, tray_length, minSize };

    // Gọi autoCountDiamonds
    box.style.border = "2px solid red"; // Hồng
    document.getElementById("confirmTrayBtn").style.display = "none";
    autoCountDiamonds();
});

// Hàm này sẽ bật/tắt các nút và làm mờ ảnh preview
function setUIEnabled(enabled) {
    // Danh sách id các nút cần disable/enable
    const ids = [
        "fileInput", "enhanceBtn", "autoCutBtn", "openCameraBtn", "takePhotoBtn",
        "calculateBtn", "btnWithBox", "btnNumberOnly", "partSizeSelect"
    ];
    ids.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.disabled = !enabled;
    });
    // Làm mờ ảnh review
    const preview = document.getElementById("canvasUpload");
    if (preview) {
        if (!enabled) preview.classList.add("disabled-overlay");
        else preview.classList.remove("disabled-overlay");
    }
}

//XỬ LÝ NÂNG CẤP ẢNH
// Khi người dùng nhấn nút "Nâng cấp ảnh", sẽ làm sắc nét ảnh
// Hàm này sẽ làm sắc nét ảnh bằng bộ lọc
// Sử dụng bộ lọc sắc nét đơn giản
function enhanceImage() {
    const canvas = document.getElementById("canvasUpload");
    const ctx = canvas.getContext("2d");
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    const kernel = [
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0
    ];

    const copy = new Uint8ClampedArray(data); // Bản sao dữ liệu gốc
    const w = canvas.width, h = canvas.height;

    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            for (let c = 0; c < 3; c++) {
                let i = (y * w + x) * 4 + c;
                let val = 0;

                for (let ky = -1; ky <= 1; ky++) {
                    for (let kx = -1; kx <= 1; kx++) {
                        const weight = kernel[(ky + 1) * 3 + (kx + 1)];
                        const ni = ((y + ky) * w + (x + kx)) * 4 + c;
                        val += copy[ni] * weight;
                    }
                }

                data[i] = Math.min(255, Math.max(0, val));
            }
        }
    }

    ctx.putImageData(imageData, 0, 0);

}


// Xử lý sự kiện khi người dùng nhấn nút "Nâng cấp ảnh"
document.getElementById("enhanceBtn").addEventListener("click", enhanceImage);

// Hiển thị giá trị font size và min size khi người dùng thay đổi
document.getElementById("fontSize").addEventListener("input", function () {
    document.getElementById("fontSizeValue").innerText = this.value;
});

// Hiển thị giá trị min size khi người dùng thay đổi
document.getElementById("minSize").addEventListener("input", function () {
    document.getElementById("minSizeValue").innerText = this.value;
});

// Hàm này sẽ resize và pad ảnh về kích thước 640x640
// Nó sẽ giữ nguyên tỷ lệ ảnh gốc và thêm viền đen nếu cần
const TARGET_SIZE = 640;
function resizeAndPadImage(file, targetSize) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = function (e) {
            const img = new Image();
            img.onload = function () {
                const ratio = Math.min(targetSize / img.width, targetSize / img.height);
                const newWidth = Math.round(img.width * ratio);
                const newHeight = Math.round(img.height * ratio);

                const canvas = document.getElementById("canvasUpload");
                canvas.width = targetSize;
                canvas.height = targetSize;
                canvas.style.width = "320px";
                const ctx = canvas.getContext("2d");

                ctx.fillStyle = "black";
                ctx.fillRect(0, 0, targetSize, targetSize);

                const xOffset = Math.floor((targetSize - newWidth) / 2);
                const yOffset = Math.floor((targetSize - newHeight) / 2);
                ctx.drawImage(img, xOffset, yOffset, newWidth, newHeight);

                canvas.toBlob((blob) => {
                    resolve(blob);
                }, file.type);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    });
}


// Hàm này sẽ gửi ảnh đã chọn lên server để xử lý
// Nó sẽ gửi ảnh đã resize và pad, cùng với các thông tin khác như vùng chọn
async function sendImageToServer(mode) {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];
    if (!file) {
        alert("Vui lòng chọn ảnh trước.");
        return;
    }

    const canvas = document.getElementById("canvasUpload");
    const processedBlob = await new Promise(resolve =>
        canvas.toBlob(blob => resolve(blob), file.type)
    );
    const formData = new FormData();

    const scaleFactor = TARGET_SIZE / 320;
    const boxX = Math.round(box.offsetLeft * scaleFactor);
    const boxY = Math.round(box.offsetTop * scaleFactor);
    const boxW = Math.round(box.offsetWidth * scaleFactor);
    const boxH = Math.round(box.offsetHeight * scaleFactor);

    // Gửi thêm thông tin vùng chọn
    formData.append("region_x", boxX);
    formData.append("region_y", boxY);
    formData.append("region_w", boxW);
    formData.append("region_h", boxH);


    const fontSize = parseInt(document.getElementById("fontSize").value);
    formData.append("font_size", fontSize);


    const minSize = parseFloat(document.getElementById("minSize").value);
    formData.append("min_size", minSize);

    const circleThreshold = parseFloat(document.getElementById("circleThreshold").value);
    formData.append("circle_threshold", circleThreshold);



    formData.append("image", processedBlob, file.name);
    const trayLength = document.getElementById("trayLength").value;
    const trayWidth = document.getElementById("trayWidth").value;

    formData.append("tray_length", trayLength);
    formData.append("tray_width", trayWidth);
    try {
        const res = await fetch(`/predict?mode=${mode}`, {
            method: "POST",
            body: formData,
        });

        const data = await res.json();
        document.getElementById("totalCountDisplay").innerText = "Tổng số: " + data.diamond_count + "💎";
        console.log("chieudai_khay + chieurong_khay hihi:", data.chieudai_khay, data.chieurong_khay);
        if (data.annotated_image) {
            const img = new Image();
            img.onload = () => originalAnnotatedImageOnload(img);
            img.src = "data:image/jpeg;base64," + data.annotated_image;
        }

        const tbody = document.getElementById("resultsTableBody");
        tbody.innerHTML = "";
        if (Array.isArray(data.details)) {
            data.details.forEach((item) => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
    <td style="padding-left: 20px;">${item.index}</td>
    <td style="padding-left: 20px;">${item.width.toFixed(2)}</td>
    <td style="padding-left: 20px;">${item.height.toFixed(2)}</td>
    <td style="padding-left: 20px;">${item.shape}</td>
    <td style="padding-left: 20px;">${item.area.toFixed(2)}</td>
    <td style="padding-left: 20px;">${item.diameter.toFixed(2)}</td>
  `;
                tbody.appendChild(tr);
            });
        }
    } catch (error) {
        alert("Lỗi khi gửi ảnh lên server.");
        console.error(error);
    }
}


// Xử lý sự kiện khi người dùng chọn ảnh từ máy tính
// Khi người dùng chọn ảnh, sẽ resize và pad ảnh về kích thước 640x640
document.getElementById("fileInput").addEventListener("change", async function () {
    const file = this.files[0];
    if (file) {
        await resizeAndPadImage(file, TARGET_SIZE);
        showResizableBox();
    }
});


// Xử lý sự kiện khi người dùng nhấn nút "Tự động đếm kim cương"
document.getElementById("btnWithBox").addEventListener("click", () => sendImageToServer("box"));


// Hiển thị khung resizable
// Khung này sẽ cho phép người dùng kéo thả để chọn vùng cần đếm kim
let box = document.getElementById("resizableBox");

// Thiết lập canvas để hiển thị ảnh đã chọn
// Canvas này sẽ hiển thị ảnh đã resize và pad, cùng với vùng chọn
let canvas = document.getElementById("canvasUpload");


// Hiển thị khung resizable khi người dùng chọn ảnh
// Hàm này sẽ hiển thị khung resizable và đặt vị trí, kích thước ban đầu
// Khung này sẽ cho phép người dùng kéo thả để chọn vùng cần đếm kim cương
function showResizableBox() {
    box.style.display = "block";
    box.style.left = "20px";
    box.style.top = "20px";
    box.style.width = "100px";
    box.style.height = "100px";
}

let isDragging = false;
let offsetX = 0, offsetY = 0;

// Kéo di chuyển khung bằng cảm ứng
box.addEventListener("touchstart", function (e) {
    if (e.target.classList.contains("resizer")) return;
    isDragging = true;
    const touch = e.touches[0];
    const rect = box.getBoundingClientRect();
    offsetX = touch.clientX - rect.left;
    offsetY = touch.clientY - rect.top;
    e.preventDefault();
}, { passive: false });

document.addEventListener("touchmove", function (e) {
    if (!isDragging) return;
    const touch = e.touches[0];
    let previewRect = canvas.getBoundingClientRect();
    let x = touch.clientX - previewRect.left - offsetX;
    let y = touch.clientY - previewRect.top - offsetY;

    x = Math.max(0, Math.min(x, 320 - box.offsetWidth));
    y = Math.max(0, Math.min(y, 320 - box.offsetHeight));

    box.style.left = `${x}px`;
    box.style.top = `${y}px`;
    e.preventDefault();
}, { passive: false });

document.addEventListener("touchend", () => isDragging = false);

// Resize 4 góc bằng cảm ứng
const resizers = box.querySelectorAll(".resizer");
let isResizing = false;
let currentResizer;

resizers.forEach(resizer => {
    resizer.addEventListener("touchstart", function (e) {
        isResizing = true;
        currentResizer = resizer;
        e.preventDefault();
        e.stopPropagation();
    }, { passive: false });
});

document.addEventListener("touchmove", function (e) {
    if (!isResizing) return;
    const touch = e.touches[0];
    const canvasRect = canvas.getBoundingClientRect();
    const boxRect = box.getBoundingClientRect();

    const dx = touch.clientX - boxRect.left;
    const dy = touch.clientY - boxRect.top;

    let width = box.offsetWidth;
    let height = box.offsetHeight;
    let left = box.offsetLeft;
    let top = box.offsetTop;

    if (currentResizer.classList.contains("bottom-right")) {
        width = dx;
        height = dy;
    } else if (currentResizer.classList.contains("bottom-left")) {
        width = box.offsetWidth - (touch.clientX - boxRect.left);
        left += (touch.clientX - boxRect.left);
        height = dy;
    } else if (currentResizer.classList.contains("top-right")) {
        height = box.offsetHeight - (touch.clientY - boxRect.top);
        top += (touch.clientY - boxRect.top);
        width = dx;
    } else if (currentResizer.classList.contains("top-left")) {
        width = box.offsetWidth - (touch.clientX - boxRect.left);
        height = box.offsetHeight - (touch.clientY - boxRect.top);
        left += (touch.clientX - boxRect.left);
        top += (touch.clientY - boxRect.top);
    }

    // Giới hạn để khung không tràn ra ngoài
    left = Math.max(0, Math.min(left, 320 - 20));
    top = Math.max(0, Math.min(top, 320 - 20));
    width = Math.max(20, Math.min(width, 320 - left));
    height = Math.max(20, Math.min(height, 320 - top));

    box.style.width = `${width}px`;
    box.style.height = `${height}px`;
    box.style.left = `${left}px`;
    box.style.top = `${top}px`;
    e.preventDefault();
}, { passive: false });

document.addEventListener("touchend", () => isResizing = false);

// Resize bằng chuột
resizers.forEach(resizer => {
    resizer.addEventListener("mousedown", function (e) {
        isResizing = true;
        currentResizer = resizer;
        e.preventDefault();
        e.stopPropagation();
    });
});

document.addEventListener("mousemove", function (e) {
    if (!isResizing) return;
    const canvasRect = canvas.getBoundingClientRect();
    const boxRect = box.getBoundingClientRect();

    const dx = e.clientX - boxRect.left;
    const dy = e.clientY - boxRect.top;

    let width = box.offsetWidth;
    let height = box.offsetHeight;
    let left = box.offsetLeft;
    let top = box.offsetTop;

    if (currentResizer.classList.contains("bottom-right")) {
        width = dx;
        height = dy;
    } else if (currentResizer.classList.contains("bottom-left")) {
        width = box.offsetWidth - (e.clientX - boxRect.left);
        left += (e.clientX - boxRect.left);
        height = dy;
    } else if (currentResizer.classList.contains("top-right")) {
        height = box.offsetHeight - (e.clientY - boxRect.top);
        top += (e.clientY - boxRect.top);
        width = dx;
    } else if (currentResizer.classList.contains("top-left")) {
        width = box.offsetWidth - (e.clientX - boxRect.left);
        height = box.offsetHeight - (e.clientY - boxRect.top);
        left += (e.clientX - boxRect.left);
        top += (e.clientY - boxRect.top);
    }

    // Giới hạn để khung không tràn ra ngoài
    left = Math.max(0, Math.min(left, 320 - 20));
    top = Math.max(0, Math.min(top, 320 - 20));
    width = Math.max(20, Math.min(width, 320 - left));
    height = Math.max(20, Math.min(height, 320 - top));

    box.style.width = `${width}px`;
    box.style.height = `${height}px`;
    box.style.left = `${left}px`;
    box.style.top = `${top}px`;
});

document.addEventListener("mouseup", () => isResizing = false);

// XỬ LÝ MỞ CAMERA VÀ CHỤP ẢNH
// Biến toàn cục để lưu trữ stream video và trạng thái camera
let videoStream = null;
let currentFacingMode = "environment";

const video = document.getElementById("video");
const cameraSelect = document.getElementById("cameraSelect");
const openCameraBtn = document.getElementById("openCameraBtn");
const takePhotoBtn = document.getElementById("takePhotoBtn");
const capturedPhoto = document.getElementById("capturedPhoto");
const videoContainer = document.querySelector(".video-container");
const uploadTabBtn = document.querySelector('[data-tab="uploadTab"]');
const captureTabBtn = document.querySelector('[data-tab="captureTab"]');
const uploadTab = document.getElementById("uploadTab");
const captureTab = document.getElementById("captureTab");

// Tab switching logic
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', function () {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        if (this.dataset.tab === 'uploadTab') {
            uploadTab.classList.add('active');
            captureTab.classList.remove('active');
        } else if (this.dataset.tab === 'captureTab') {
            uploadTab.classList.remove('active');
            captureTab.classList.add('active');
            // Nếu là thiết bị di động thì tự động mở input file với capture
            if (/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {
                const mobileCaptureInput = document.getElementById('mobileCaptureInput');
                if (mobileCaptureInput) {
                    mobileCaptureInput.value = '';
                    mobileCaptureInput.click();
                }
            }
        }
    });
});

// Mở camera
openCameraBtn.addEventListener("click", async function () {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
    }
    currentFacingMode = cameraSelect.value;

    try {
        // Thử với exact facing mode trước
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { exact: currentFacingMode },
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });
    } catch (e) {
        console.log("Không thể mở camera với exact mode, thử mode thường:", e);
        try {
            // Thử với facing mode thường
            videoStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: currentFacingMode,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });
        } catch (e2) {
            console.log("Không thể mở camera, thử camera bất kỳ:", e2);
            // Thử với camera bất kỳ
            videoStream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: false
            });
        }
    }

    if (videoStream) {
        video.srcObject = videoStream;
        videoContainer.style.display = "block";
        takePhotoBtn.style.display = "inline-block";
        capturedPhoto.style.display = "none";

        // Đảm bảo video load xong
        video.onloadedmetadata = function () {
            console.log("Camera đã sẵn sàng");
        };
    } else {
        alert("Không thể mở camera. Vui lòng kiểm tra quyền truy cập camera.");
    }
});

// CHỤP ẢNH TỪ CAMERA
// Khi người dùng nhấn nút "Chụp ảnh", sẽ chụp ảnh
takePhotoBtn.addEventListener("click", function () {
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const ctx = tempCanvas.getContext("2d");
    ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    const dataUrl = tempCanvas.toDataURL("image/png");
    capturedPhoto.src = dataUrl;
    capturedPhoto.style.display = "block";
    // Chuyển ảnh sang tab upload và vẽ lên canvasUpload
    uploadTabBtn.click();
    // Resize và pad ảnh như upload
    const img = new Image();
    img.onload = async function () {
        const ratio = Math.min(TARGET_SIZE / img.width, TARGET_SIZE / img.height);
        const newWidth = Math.round(img.width * ratio);
        const newHeight = Math.round(img.height * ratio);
        const canvasUpload = document.getElementById("canvasUpload");
        canvasUpload.width = TARGET_SIZE;
        canvasUpload.height = TARGET_SIZE;
        canvasUpload.style.width = "320px";
        const ctx2 = canvasUpload.getContext("2d");
        ctx2.fillStyle = "black";
        ctx2.fillRect(0, 0, TARGET_SIZE, TARGET_SIZE);
        const xOffset = Math.floor((TARGET_SIZE - newWidth) / 2);
        const yOffset = Math.floor((TARGET_SIZE - newHeight) / 2);
        ctx2.drawImage(img, xOffset, yOffset, newWidth, newHeight);
        showResizableBox();
        // Tạo file từ ảnh vừa chụp và gán vào fileInput
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            const res = await fetch(dataUrl);
            const blob = await res.blob();
            const file = new File([blob], 'captured_photo.png', { type: 'image/png' });
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
        }
    };
    img.src = dataUrl;
});

// Khi người dùng chọn camera, sẽ thay đổi facing mode
captureTabBtn.addEventListener('click', function () {
    // Không tự động hiển thị video container khi chuyển tab
    // Chỉ hiển thị khi người dùng nhấn "Mở Camera"
    videoContainer.style.display = "none";
    takePhotoBtn.style.display = "none";
});

// Khi người dùng chuyển sang tab upload, sẽ ẩn video và nút chụp ảnh
uploadTabBtn.addEventListener('click', function () {
    videoContainer.style.display = "none";
    takePhotoBtn.style.display = "none";
    // Dừng camera nếu đang chạy
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
});

// Hiển thị danh sách camera
function showResultLoadingOverlay() {
    document.getElementById('resultLoadingOverlay').style.display = 'flex';
}
function hideResultLoadingOverlay() {
    document.getElementById('resultLoadingOverlay').style.display = 'none';
}
// Hook vào autoCountDiamonds và sendImageToServer
const _autoCountDiamonds = autoCountDiamonds;
autoCountDiamonds = async function (...args) {
    showResultLoadingOverlay();
    try {
        await _autoCountDiamonds.apply(this, args);
    } finally {
        hideResultLoadingOverlay();
    }
}

// Hook vào sendImageToServer để hiển thị overlay loading
// Hàm này sẽ hiển thị overlay loading khi gửi ảnh lên server
const _sendImageToServer = sendImageToServer;
sendImageToServer = async function (...args) {
    showResultLoadingOverlay();
    try {
        await _sendImageToServer.apply(this, args);
    } finally {
        hideResultLoadingOverlay();
    }
}

// Ngăn chặn sự kiện beforeunload để tránh mất dữ liệu
// Điều này sẽ hiển thị hộp thoại xác nhận khi người dùng cố gắng đóng
window.addEventListener("beforeunload", function (e) {
    e.preventDefault();
});

// ✅ Gán biến toàn cục để tránh lỗi ReferenceError
let shapeChart = null;


// Xử lý sự kiện mở modal thống kê
// Khi người dùng nhấn nút "Mở Dashboard", sẽ hiển thị modal thống
document.addEventListener("DOMContentLoaded", () => {
    const modal = document.getElementById("statsModal");
    const closeBtn = modal.querySelector(".close");
    const openBtn = document.getElementById("openDashboardBtn");

    openBtn.addEventListener("click", () => {
        const rows = document.querySelectorAll("#resultsTableBody tr");
        let total = 0, circle = 0, triangle = 0;

        rows.forEach(row => {
            const cells = row.querySelectorAll("td");
            if (cells.length >= 4) {
                total++;
                const shape = cells[3].innerText.trim();
                if (shape === "Tròn") circle++;
                else if (shape === "Tam giác") triangle++;
            }
        });

        document.getElementById("statTotal").innerText = total;
        document.getElementById("statCircle").innerText = circle;
        document.getElementById("statTriangle").innerText = triangle;

        if (shapeChart) shapeChart.destroy();

        const ctx = document.getElementById("shapeChart").getContext("2d");
        shapeChart = new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: ["Hình Tròn", "Tam Giác"],
                datasets: [{
                    data: [circle, triangle],
                    backgroundColor: ["#007bff", "#dc3545"]
                }]
            }
        });

        modal.style.display = "block";
    });

    closeBtn.onclick = () => modal.style.display = "none";
    window.onclick = (e) => {
        if (e.target === modal) modal.style.display = "none";
    };
});

function updateMinSizeSliderBg() {
    const slider = document.getElementById('minSize');
    const min = parseFloat(slider.min);
    const max = parseFloat(slider.max);
    const val = parseFloat(slider.value);
    const percent = ((val - min) / (max - min)) * 100;
    slider.style.background = `linear-gradient(to right, #E6266F 0%, #E6266F ${percent}%, #eee ${percent}%, #eee 100%)`;
}
document.getElementById('minSize').addEventListener('input', updateMinSizeSliderBg);
window.addEventListener('DOMContentLoaded', updateMinSizeSliderBg);

function updateSliderBg(sliderId) {
    const slider = document.getElementById(sliderId);
    if (!slider) return;
    const min = parseFloat(slider.min);
    const max = parseFloat(slider.max);
    const val = parseFloat(slider.value);
    const percent = ((val - min) / (max - min)) * 100;
    slider.style.background = `linear-gradient(to right, #E6266F 0%, #E6266F ${percent}%, #eee ${percent}%, #eee 100%)`;
}

// Gọi cho cả hai slider khi load và khi thay đổi
['minSize', 'fontSize'].forEach(id => {
    const slider = document.getElementById(id);
    if (slider) {
        slider.addEventListener('input', () => updateSliderBg(id));
        // Gọi 1 lần khi load trang
        updateSliderBg(id);
    }
});

window.addEventListener('DOMContentLoaded', function () {
    var cutGroup = document.getElementById('cutOptionsGroup');
    if (cutGroup) {
        cutGroup.classList.remove('show');
        cutGroup.style.display = 'none';
    }
});

// Cập nhật giá trị hiển thị cho slider circleThreshold
const circleThresholdSlider = document.getElementById('circleThreshold');
const circleThresholdValue = document.getElementById('circleThresholdValue');
if (circleThresholdSlider && circleThresholdValue) {
    circleThresholdSlider.addEventListener('input', function () {
        circleThresholdValue.textContent = this.value;
    });
}

const circleSlider = document.getElementById('circleThreshold');
if (circleSlider) {
    updateSliderBg('circleThreshold');
    circleSlider.addEventListener('input', function () {
        updateSliderBg('circleThreshold');
    });
}

// Khi nhấn tab "Tải lên hình ảnh" trên mobile, tự động click input file
let userClickedUploadTab = false;

// Đánh dấu khi người dùng thực sự bấm vào tab upload
const uploadTabBtnEl = document.querySelector('[data-tab="uploadTab"]');
uploadTabBtnEl.addEventListener('mousedown', function () {
    userClickedUploadTab = true;
});
uploadTabBtnEl.addEventListener('touchstart', function () {
    userClickedUploadTab = true;
});

uploadTabBtnEl.addEventListener('click', function () {
    if (window.innerWidth <= 600) {
        if (userClickedUploadTab) {
            document.getElementById('fileInput').click();
        }
    }
    userClickedUploadTab = false;
});

// Hiển thị modal lưu ý khi nhấn chấm hỏi
const helpIcon = document.getElementById('helpIcon');
const helpModal = document.getElementById('helpModal');
const closeHelpModal = document.getElementById('closeHelpModal');
if (helpIcon && helpModal && closeHelpModal) {
    helpIcon.addEventListener('click', function () {
        helpModal.style.display = 'block';
    });
    closeHelpModal.addEventListener('click', function () {
        helpModal.style.display = 'none';
    });
    window.addEventListener('click', function (e) {
        if (e.target === helpModal) helpModal.style.display = 'none';
    });
}

// Xử lý khi người dùng chọn/chụp ảnh từ input file mobileCaptureInput
const mobileCaptureInput = document.getElementById('mobileCaptureInput');
if (mobileCaptureInput) {
    mobileCaptureInput.addEventListener('change', async function () {
        const file = this.files[0];
        if (file) {
            await resizeAndPadImage(file, TARGET_SIZE);
            showResizableBox();
            // Gán file vừa chụp vào fileInput để nút Tính Toán hoạt động
            const fileInput = document.getElementById('fileInput');
            if (fileInput) {
                // Tạo FileList mới từ file vừa chụp
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                fileInput.files = dataTransfer.files;
            }
            // Tự động chuyển về tab upload
            if (uploadTabBtn) uploadTabBtn.click();
        }
    });
}


// XỬ LÝ PHÓNG TO/THU NHỎ VÀ DI CHUYỂN ẢNH KẾT QUẢ
const zoomContainer = document.getElementById('zoomContainer');
const canvasAnnotated = document.getElementById('canvasAnnotated');
const ctxAnnotated = canvasAnnotated.getContext('2d');

let scale = 1;
let translateX = 0;
let translateY = 0;
let isPanning = false;
let startX = 0;
let startY = 0;

// Ghi đè hàm onload của ảnh để vẽ lại với các thuộc tính transform
const originalAnnotatedImageOnload = (img) => {
    canvasAnnotated.width = img.width;
    canvasAnnotated.height = img.height;
    // Đảm bảo canvas luôn vừa với khung hiển thị ban đầu
    canvasAnnotated.style.width = "100%";
    canvasAnnotated.style.height = "100%";

    // Đặt lại các giá trị zoom/pan khi ảnh mới được tải
    scale = 1;
    translateX = 0;
    translateY = 0;
    updateTransform();

    ctxAnnotated.drawImage(img, 0, 0);
};

// Cập nhật lại logic tải ảnh trong hàm autoCountDiamonds và sendImageToServer
// để sử dụng hàm onload đã ghi đè
// Tìm và thay thế img.onload trong autoCountDiamonds
// Tìm và thay thế img.onload trong sendImageToServer

function updateTransform() {
    canvasAnnotated.style.transformOrigin = `0 0`; // Set transform origin to top-left
    canvasAnnotated.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
}

zoomContainer.addEventListener('wheel', (e) => {
    e.preventDefault();
    const scaleAmount = 0.1;
    const mouseX = e.clientX - zoomContainer.getBoundingClientRect().left;
    const mouseY = e.clientY - zoomContainer.getBoundingClientRect().top;

    const oldScale = scale;
    if (e.deltaY < 0) {
        scale += scaleAmount; // Zoom in
    } else {
        scale -= scaleAmount; // Zoom out
    }
    scale = Math.max(0.5, Math.min(scale, 5)); // Limit zoom level

    // Adjust translation to zoom towards the mouse pointer
    translateX = mouseX - (mouseX - translateX) * (scale / oldScale);
    translateY = mouseY - (mouseY - translateY) * (scale / oldScale);

    updateTransform();
});

zoomContainer.addEventListener('mousedown', (e) => {
    if (scale > 1) { // Only allow panning if zoomed in
        isPanning = true;
        zoomContainer.style.cursor = 'grabbing';
        startX = e.clientX - translateX;
        startY = e.clientY - translateY;
    }
});

zoomContainer.addEventListener('mousemove', (e) => {
    if (!isPanning) return;
    translateX = e.clientX - startX;
    translateY = e.clientY - startY;

    // Constrain panning to keep the image within the container
    const scaledWidth = canvasAnnotated.width * scale;
    const scaledHeight = canvasAnnotated.height * scale;
    const containerWidth = zoomContainer.clientWidth;
    const containerHeight = zoomContainer.clientHeight;

    // Prevent panning too far left
    translateX = Math.min(translateX, 0);
    // Prevent panning too far right
    translateX = Math.max(translateX, containerWidth - scaledWidth);
    // Prevent panning too far up
    translateY = Math.min(translateY, 0);
    // Prevent panning too far down
    translateY = Math.max(translateY, containerHeight - scaledHeight);

    updateTransform();
});

zoomContainer.addEventListener('mouseup', () => {
    isPanning = false;
    zoomContainer.style.cursor = 'grab';
});

zoomContainer.addEventListener('mouseleave', () => {
    isPanning = false;
});

// ACTIVE LEARNING FEEDBACK FUNCTIONS
function enableFeedbackMode() {
    const canvas = document.getElementById("canvasAnnotated"); // Đổi sang canvas kết quả có số
    if (!canvas) {
        showToast("❌ Vui lòng tính toán trước khi cải thiện model!", "error");
        return;
    }
    
    // Remove existing overlay if any
    const existingOverlay = document.getElementById('feedback-overlay');
    if (existingOverlay) {
        existingOverlay.remove();
    }
    
    const existingControls = document.getElementById('feedback-controls');
    if (existingControls) {
        existingControls.remove();
    }
    
    const feedbackOverlay = document.createElement('div');
    feedbackOverlay.id = 'feedback-overlay';
    feedbackOverlay.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: auto;
        z-index: 1000;
        cursor: crosshair;
        background: rgba(255, 215, 0, 0.15);
        border: 3px solid #FFD700;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.8);
        animation: feedbackPulse 2s infinite;
    `;
    
    canvas.parentElement.appendChild(feedbackOverlay);
    
    // Add improved feedback controls
    const feedbackControls = document.createElement('div');
    feedbackControls.innerHTML = `
        <div style="position: fixed; top: 10px; right: 10px; background: white; padding: 12px; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); z-index: 1001; max-width: 200px; font-size: 12px;">
            <h4 style="margin: 0 0 8px 0; color: #333; font-size: 14px;">🎯 Cải Thiện Model</h4>
            
            <div style="background: #fff3cd; border: 1px solid #ffc107; padding: 8px; border-radius: 5px; margin-bottom: 8px; font-size: 11px;">
                <strong style="color: #856404;">📍 Hướng dẫn:</strong><br>
                🔴 Click <strong>SỐ VÀNG</strong> nếu SAI<br>
                🟠 Click <strong>CHỖ TRỐNG</strong> nếu THIẾU<br>
                <em style="color: #6c757d;">💡 Click trên ảnh có số!</em>
            </div>
            
            <div style="margin-bottom: 8px; font-size: 11px;">
                <div>✅ Đúng: <span id="tpCount" style="font-weight: bold; color: green;">0</span></div>
                <div>❌ Sai: <span id="fpCount" style="font-weight: bold; color: red;">0</span></div>
                <div>➕ Thiếu: <span id="missedCount" style="font-weight: bold; color: orange;">0</span></div>
            </div>
            
            <div style="display: flex; gap: 5px; margin-bottom: 5px;">
                <button onclick="submitFeedback()" style="flex: 1; background: #4CAF50; color: white; border: none; padding: 6px; border-radius: 3px; cursor: pointer; font-weight: bold; font-size: 10px;">
                    ✅ Gửi
                </button>
                <button onclick="cancelFeedback()" style="flex: 1; background: #f44336; color: white; border: none; padding: 6px; border-radius: 3px; cursor: pointer; font-size: 10px;">
                    ❌ Hủy
                </button>
            </div>
            
            <div style="border-top: 1px solid #ffc107; padding-top: 5px; margin-top: 5px;">
                <button onclick="startRetraining()" style="width: 100%; background: linear-gradient(135deg, #ff6b35, #f7931e); color: white; border: none; padding: 6px; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 10px; margin-bottom: 3px;">
                    🚀 Train Model
                </button>
                <button onclick="reloadModel()" style="width: 100%; background: linear-gradient(135deg, #28a745, #20c997); color: white; border: none; padding: 6px; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 10px;">
                    🔄 Load New Model
                </button>
                <div id="retrainStatus" style="margin-top: 4px; font-size: 9px; text-align: center;"></div>
            </div>
            
            <div style="margin-top: 5px; font-size: 9px; color: #666; text-align: center;">
                💡 Feedback giúp model học tốt hơn
            </div>
        </div>
    `;
    document.body.appendChild(feedbackControls);
    feedbackControls.id = 'feedback-controls';
    
    // Add click handlers
    feedbackOverlay.addEventListener('click', handleFeedbackClick);
    
    // Update counters
    updateFeedbackCounters();
}

function handleFeedbackClick(event) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Check if clicking on existing detection
    const clickedDetection = findDetectionAtPoint(x, y);
    
    if (clickedDetection) {
        // Mark as false positive (detection sai)
        if (!window.userCorrections.false_positives.includes(clickedDetection.index)) {
            window.userCorrections.false_positives.push(clickedDetection.index);
            highlightFalsePositive(clickedDetection);
            
            // Show feedback to user
            showToast(`❌ Đã đánh dấu detection #${clickedDetection.index + 1} là SAI`, 'error');
        } else {
            // Unmark false positive
            const index = window.userCorrections.false_positives.indexOf(clickedDetection.index);
            window.userCorrections.false_positives.splice(index, 1);
            removeFalsePositiveHighlight(clickedDetection.index);
            showToast(`↩️ Đã bỏ đánh dấu detection #${clickedDetection.index + 1}`, 'info');
        }
    } else {
        // Mark as missed object (thiếu kim cương)
        const missedObject = {
            x: x - 25, y: y - 25, w: 50, h: 50,
            description: `Missed diamond at (${Math.round(x)}, ${Math.round(y)})`
        };
        window.userCorrections.missed_objects.push(missedObject);
        highlightMissedObject(missedObject, window.userCorrections.missed_objects.length - 1);
        
        showToast(`➕ Đã thêm kim cương thiếu tại (${Math.round(x)}, ${Math.round(y)})`, 'success');
    }
    
    // Update counters
    updateFeedbackCounters();
}

function findDetectionAtPoint(x, y) {
    // Find detection box at clicked point
    for (let i = 0; i < window.currentPredictions.length; i++) {
        const pred = window.currentPredictions[i];
        if (pred.x <= x && x <= pred.x + pred.w && 
            pred.y <= y && y <= pred.y + pred.h) {
            return { ...pred, index: i };
        }
    }
    return null;
}

function highlightFalsePositive(detection) {
    const highlight = document.createElement('div');
    highlight.style.cssText = `
        position: absolute;
        left: ${detection.x}px;
        top: ${detection.y}px;
        width: ${detection.w}px;
        height: ${detection.h}px;
        border: 3px solid red;
        background: rgba(255, 0, 0, 0.2);
        pointer-events: none;
        z-index: 999;
    `;
    highlight.className = 'false-positive-highlight';
    document.getElementById('feedback-overlay').appendChild(highlight);
}

function highlightMissedObject(missed, index) {
    const highlight = document.createElement('div');
    // Giảm kích thước marker xuống 60% so với vùng thực tế
    const markerSize = Math.min(missed.w, missed.h) * 0.6;
    const offsetX = (missed.w - markerSize) / 2;
    const offsetY = (missed.h - markerSize) / 2;
    
    highlight.style.cssText = `
        position: absolute;
        left: ${missed.x + offsetX}px;
        top: ${missed.y + offsetY}px;
        width: ${markerSize}px;
        height: ${markerSize}px;
        border: 2px solid orange;
        background: rgba(255, 165, 0, 0.3);
        pointer-events: none;
        z-index: 999;
        border-radius: 50%;
    `;
    highlight.className = 'missed-object-highlight';
    highlight.dataset.index = index;
    
    // Add label nhỏ gọn hơn
    const label = document.createElement('div');
    label.style.cssText = `
        position: absolute;
        top: -20px;
        left: 50%;
        transform: translateX(-50%);
        background: orange;
        color: white;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 10px;
        font-weight: bold;
        white-space: nowrap;
    `;
    label.textContent = `+${index + 1}`;
    highlight.appendChild(label);
    
    document.getElementById('feedback-overlay').appendChild(highlight);
}

function removeFalsePositiveHighlight(detectionIndex) {
    const highlights = document.querySelectorAll('.false-positive-highlight');
    highlights.forEach(highlight => {
        if (highlight.dataset.index == detectionIndex) {
            highlight.remove();
        }
    });
}

function updateFeedbackCounters() {
    const tpCount = document.getElementById('tpCount');
    const fpCount = document.getElementById('fpCount');
    const missedCount = document.getElementById('missedCount');
    
    // Tính true positives từ total predictions - false positives
    const totalPredictions = window.currentPredictions.length;
    const falsePositiveCount = window.userCorrections.false_positives.length;
    const truePositiveCount = totalPredictions - falsePositiveCount;
    
    if (tpCount) tpCount.textContent = truePositiveCount;
    if (fpCount) fpCount.textContent = falsePositiveCount;
    if (missedCount) missedCount.textContent = window.userCorrections.missed_objects.length;
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: ${type === 'error' ? '#f44336' : type === 'success' ? '#4CAF50' : '#2196F3'};
        color: white;
        padding: 12px 24px;
        border-radius: 25px;
        font-weight: bold;
        z-index: 10000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        animation: slideDown 0.3s ease;
    `;
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'slideUp 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

async function submitFeedback() {
    const fpCount = window.userCorrections.false_positives.length;
    const missedCount = window.userCorrections.missed_objects.length;
    
    // Tính true positives: tất cả predictions trừ đi false positives
    const totalPredictions = window.currentPredictions.length;
    const truePositiveCount = totalPredictions - fpCount;
    
    if (fpCount === 0 && missedCount === 0) {
        alert('⚠️ Chưa có feedback nào!\n\nHãy:\n- Click vào detection SAI để đánh dấu\n- Click vào chỗ THIẾU kim cương để thêm');
        return;
    }
    
    const confirmMessage = `🎯 Xác nhận gửi feedback:\n\n` +
        `✅ Detection đúng: ${truePositiveCount}\n` +
        `❌ Detection sai: ${fpCount}\n` +
        `➕ Kim cương thiếu: ${missedCount}\n\n` +
        `Tổng cộng: ${truePositiveCount + missedCount} kim cương thực tế\n\n` +
        `Dữ liệu này sẽ giúp cải thiện model. Tiếp tục?`;
    
    if (!confirm(confirmMessage)) {
        return;
    }
    
    try {
        // Show loading
        showToast('📤 Đang gửi feedback...', 'info');
        
        // Get current image data
        const canvas = document.getElementById("canvasUpload");
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        
        // Tạo danh sách true positives (các detection không bị đánh dấu false positive)
        const truePositives = window.currentPredictions.filter((pred, index) => 
            !window.userCorrections.false_positives.includes(index)
        );
        
        // Submit feedback với đầy đủ thông tin
        const response = await fetch('/submit_feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_data: imageData.split(',')[1], // Remove data:image/jpeg;base64,
                predictions: window.currentPredictions,
                corrections: {
                    ...window.userCorrections,
                    true_positives: truePositives  // Thêm true positives
                }
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            showToast('✅ Feedback đã được gửi thành công!', 'success');
            
            // Show detailed success message
            setTimeout(() => {
                alert(`🎉 Cảm ơn bạn đã góp ý!\n\n` +
                      `✅ Detection đúng: ${truePositiveCount}\n` +
                      `❌ Detection sai: ${fpCount}\n` +
                      `➕ Thiếu: ${missedCount}\n` +
                      `📊 Tổng annotations: ${truePositiveCount + missedCount}\n\n` +
                      `🤖 Model sẽ học từ TẤT CẢ dữ liệu này\n` +
                      `📈 Độ chính xác sẽ được cải thiện\n\n` +
                      `💡 Tip: Tiếp tục sử dụng và feedback để model ngày càng tốt hơn!`);
            }, 1000);
        } else {
            showToast('❌ Lỗi gửi feedback: ' + result.message, 'error');
        }
        
    } catch (error) {
        showToast('❌ Lỗi kết nối: ' + error.message, 'error');
    }
    
    cancelFeedback();
}

function cancelFeedback() {
    // Remove feedback overlay and controls
    const overlay = document.getElementById('feedback-overlay');
    const controls = document.getElementById('feedback-controls');
    
    if (overlay) overlay.remove();
    if (controls) controls.remove();
    
    // Reset corrections
    window.userCorrections = {
        false_positives: [],
        missed_objects: []
    };
}

// Add feedback button to main interface
function addFeedbackButton() {
    // Tìm container phù hợp để thêm nút
    const container = document.querySelector('.button-group') || 
                     document.querySelector('#cutOptionsGroup').parentElement || 
                     document.querySelector('.upload-capture-section');
    
    if (!container) {
        console.error('Không tìm thấy container để thêm nút feedback');
        return;
    }
    
    // Kiểm tra xem nút đã tồn tại chưa
    if (document.getElementById('feedbackBtn')) {
        return; // Đã có nút rồi
    }
    
    const feedbackBtn = document.createElement('button');
    feedbackBtn.id = 'feedbackBtn';
    feedbackBtn.innerHTML = '🎯 Cải Thiện Model';
    feedbackBtn.style.cssText = `
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 25px;
        font-weight: bold;
        cursor: pointer;
        margin: 10px 5px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        font-size: 14px;
        transition: all 0.3s ease;
        animation: pulse 2s infinite;
        display: inline-block;
        position: relative;
        z-index: 100;
    `;
    
    feedbackBtn.onmouseover = function() {
        this.style.transform = 'scale(1.05)';
        this.style.boxShadow = '0 6px 20px rgba(0,0,0,0.3)';
    };
    feedbackBtn.onmouseout = function() {
        this.style.transform = 'scale(1)';
        this.style.boxShadow = '0 4px 15px rgba(0,0,0,0.2)';
    };
    feedbackBtn.onclick = enableFeedbackMode;
    
    // Thêm vào container đầu tiên tìm được
    container.appendChild(feedbackBtn);
    
    console.log('✅ Đã thêm nút feedback vào:', container.className || container.tagName);
    
    // Add CSS animation nếu chưa có
    if (!document.getElementById('feedback-animations')) {
        const style = document.createElement('style');
        style.id = 'feedback-animations';
        style.textContent = `
            @keyframes pulse {
                0% { box-shadow: 0 4px 15px rgba(0,0,0,0.2), 0 0 0 0 rgba(255, 107, 107, 0.7); }
                70% { box-shadow: 0 4px 15px rgba(0,0,0,0.2), 0 0 0 10px rgba(255, 107, 107, 0); }
                100% { box-shadow: 0 4px 15px rgba(0,0,0,0.2), 0 0 0 0 rgba(255, 107, 107, 0); }
            }
            @keyframes slideDown {
                from { opacity: 0; transform: translateX(-50%) translateY(-20px); }
                to { opacity: 1; transform: translateX(-50%) translateY(0); }
            }
            @keyframes slideUp {
                from { opacity: 1; transform: translateX(-50%) translateY(0); }
                to { opacity: 0; transform: translateX(-50%) translateY(-20px); }
            }
        `;
        document.head.appendChild(style);
    }
}

// RETRAINING FUNCTIONS
async function startRetraining() {
    // Sử dụng element ID mới từ HTML
    const statusDiv = document.getElementById('trainStatus') || document.getElementById('retrainStatus');
    
    if (!statusDiv) {
        showToast('❌ Không tìm thấy status element', 'error');
        return;
    }
    
    try {
        // Show loading
        statusDiv.innerHTML = '🔄 Đang bắt đầu training...';
        statusDiv.style.color = '#ffc107';
        
        // Get training parameters từ user với dialog cải tiến
        const epochs = prompt('Số epochs để train (mặc định 100 cho chất lượng tốt):', '100');
        if (!epochs || isNaN(epochs)) {
            statusDiv.innerHTML = '❌ Đã hủy training';
            statusDiv.style.color = '#dc3545';
            return;
        }
        
        // Model size selection
        const modelChoice = prompt(`Chọn kích thước model:
n - Nano (nhanh, nhỏ)
s - Small (cân bằng) 
m - Medium (chính xác hơn)
l - Large (chất lượng cao)
x - X-Large (tốt nhất)

Nhập lựa chọn (mặc định 's'):`, 's');
        
        const validSizes = ['n', 's', 'm', 'l', 'x'];
        const modelSize = validSizes.includes(modelChoice?.toLowerCase()) ? modelChoice.toLowerCase() : 's';
        
        // Start retraining
        const response = await fetch('/start_retraining', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                epochs: parseInt(epochs),
                model_size: modelSize
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            statusDiv.innerHTML = `🚀 Training started: ${result.model_size}, ${result.epochs} epochs`;
            statusDiv.style.color = '#28a745';
            
            showToast(`🚀 Training Roboflow bắt đầu! Model: ${result.model_size}`, 'success');
            
            // Start checking status periodically
            checkTrainingStatus();
            
        } else {
            statusDiv.innerHTML = '❌ Lỗi: ' + result.message;
            statusDiv.style.color = '#dc3545';
            showToast('❌ ' + result.message, 'error');
        }
        
    } catch (error) {
        statusDiv.innerHTML = '❌ Lỗi kết nối';
        statusDiv.style.color = '#dc3545';
        showToast('❌ Lỗi: ' + error.message, 'error');
    }
}

async function checkTrainingStatus() {
    const statusDiv = document.getElementById('trainStatus') || document.getElementById('retrainStatus');
    
    if (!statusDiv) return;
    
    try {
        const response = await fetch('/training_status');
        const result = await response.json();
        
        if (result.status === 'completed') {
            statusDiv.innerHTML = '✅ Training hoàn thành!';
            statusDiv.style.color = '#28a745';
            showToast('🎉 Training hoàn thành! Restart app để dùng model mới.', 'success');
            
        } else if (result.status === 'running') {
            statusDiv.innerHTML = '🏋️ Training đang chạy...';
            statusDiv.style.color = '#ffc107';
            
            // Check again after 30 seconds
            setTimeout(checkTrainingStatus, 30000);
            
        } else if (result.status === 'error') {
            statusDiv.innerHTML = '❌ Training lỗi';
            statusDiv.style.color = '#dc3545';
        }
        
    } catch (error) {
        console.error('Error checking training status:', error);
    }
}

// MODEL RELOAD FUNCTION
async function reloadModel() {
    const statusDiv = document.getElementById('trainStatus') || document.getElementById('retrainStatus');
    
    if (!statusDiv) {
        showToast('❌ Không tìm thấy status element', 'error');
        return;
    }
    
    try {
        // Show loading
        statusDiv.innerHTML = '🔄 Đang reload model...';
        statusDiv.style.color = '#ffc107';
        
        const response = await fetch('/reload_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            statusDiv.innerHTML = '✅ Model reloaded!';
            statusDiv.style.color = '#28a745';
            
            let message = '🔄 Model đã được reload thành công!';
            if (result.model_info && result.model_info.training_type === 'transfer_learning') {
                message += ` (Transfer Learning Model - ${result.model_info.epochs} epochs)`;
            }
            
            showToast(message, 'success');
            
            // Show model info briefly
            setTimeout(() => {
                if (result.model_info) {
                    statusDiv.innerHTML = `Model: ${result.model_info.model_size || 'custom'} - ${result.model_info.timestamp || 'latest'}`;
                } else {
                    statusDiv.innerHTML = '✅ Sẵn sàng với model mới';
                }
            }, 3000);
            
        } else {
            statusDiv.innerHTML = '❌ Lỗi reload: ' + result.message;
            statusDiv.style.color = '#dc3545';
            showToast('❌ ' + result.message, 'error');
        }
        
    } catch (error) {
        statusDiv.innerHTML = '❌ Lỗi kết nối';
        statusDiv.style.color = '#dc3545';
        showToast('❌ Lỗi: ' + error.message, 'error');
    }
}

// MODEL INFO FUNCTIONS
async function showModelInfo() {
    try {
        const [modelResponse, statusResponse] = await Promise.all([
            fetch('/model_info'),
            fetch('/training_status')
        ]);
        
        const modelInfo = await modelResponse.json();
        const statusInfo = await statusResponse.json();
        
        let content = `
            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; max-width: 600px;">
                <h3 style="margin: 0 0 15px 0; color: #333;">🤖 Model Information</h3>
                
                <div style="background: white; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <h4 style="margin: 0 0 10px 0; color: #495057;">📦 Current Model</h4>
                    <div style="font-family: monospace; background: #e9ecef; padding: 8px; border-radius: 4px; font-size: 12px;">
                        ${modelInfo.current_model_path}
                    </div>
                    <div style="margin-top: 8px; color: #6c757d;">
                        Type: ${modelInfo.model_type === 'retrained' ? '✨ Retrained Model' : '📦 Original Model'}
                    </div>
                </div>
        `;
        
        // Thông tin retrain nếu có
        if (modelInfo.retrain_info) {
            const info = modelInfo.retrain_info;
            content += `
                <div style="background: white; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <h4 style="margin: 0 0 10px 0; color: #495057;">✨ Retrain Details</h4>
                    <div style="display: grid; grid-template-columns: 120px 1fr; gap: 8px; font-size: 14px;">
                        <strong>Timestamp:</strong> <span>${info.timestamp}</span>
                        <strong>Model Size:</strong> <span>YOLOv8${info.model_size}-seg</span>
                        <strong>Epochs:</strong> <span>${info.epochs}</span>
                        <strong>Training Data:</strong> <span>${info.training_data}</span>
                        <strong>Created:</strong> <span>${new Date(info.created_at).toLocaleString()}</span>
                    </div>
                </div>
            `;
        }
        
        // Training history
        if (statusInfo.training_runs && statusInfo.training_runs.length > 0) {
            content += `
                <div style="background: white; padding: 15px; border-radius: 8px;">
                    <h4 style="margin: 0 0 10px 0; color: #495057;">📚 Training History</h4>
                    <div style="max-height: 200px; overflow-y: auto;">
            `;
            
            statusInfo.training_runs.forEach(run => {
                const isActive = run.is_current ? '🟢' : '⚪';
                content += `
                    <div style="padding: 8px; border-bottom: 1px solid #dee2e6; display: flex; justify-content: between; align-items: center;">
                        <div style="flex: 1;">
                            <div style="font-size: 13px; font-weight: bold;">${isActive} ${run.timestamp}</div>
                            <div style="font-size: 12px; color: #6c757d;">
                                Model: ${run.model_size} | Epochs: ${run.epochs}
                            </div>
                        </div>
                        ${run.is_current ? '<span style="background: #28a745; color: white; padding: 2px 6px; border-radius: 12px; font-size: 11px;">ACTIVE</span>' : ''}
                    </div>
                `;
            });
            
            content += `
                    </div>
                </div>
            `;
        }
        
        content += `
                <div style="margin-top: 15px; text-align: center;">
                    <button onclick="closeModelInfo()" style="background: #6c757d; color: white; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer;">
                        Đóng
                    </button>
                </div>
            </div>
        `;
        
        // Show modal
        const modal = document.createElement('div');
        modal.id = 'modelInfoModal';
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 9999;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow-y: auto;
        `;
        
        modal.innerHTML = content;
        document.body.appendChild(modal);
        
    } catch (error) {
        showToast('❌ Lỗi khi lấy thông tin model: ' + error.message, 'error');
    }
}

function closeModelInfo() {
    const modal = document.getElementById('modelInfoModal');
    if (modal) {
        modal.remove();
    }
}

// Initialize feedback system when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Nút feedback đã có trong HTML, chỉ cần đảm bảo function hoạt động
    console.log('✅ Feedback system initialized');
    
    // Đảm bảo nút feedback có sẵn
    const feedbackBtn = document.getElementById('feedbackBtn');
    if (feedbackBtn) {
        console.log('✅ Feedback button found in HTML');
        // Đảm bảo onclick handler hoạt động
        feedbackBtn.onclick = enableFeedbackMode;
    } else {
        console.warn('⚠️ Feedback button not found, adding dynamically...');
        addFeedbackButton();
    }
});


