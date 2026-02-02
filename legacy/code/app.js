/* ==========================================================================
   MULTI-DISEASE LATENT FEATURE ALIGNMENT DASHBOARD JAVASCRIPT
   Interactive Data Filtering, Image Lightbox, and Sample Matrix Render
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    initLightbox();
    initSampleMatrixTable();
    initInteractiveFilters();
});

// Lightbox Zoom Functionality
function initLightbox() {
    const plotWrapper = document.getElementById('plotDisplayWrapper');
    const lightboxModal = document.getElementById('lightboxModal');
    const closeBtn = document.getElementById('modalCloseBtn');
    const viewBtn = document.getElementById('viewHighResBtn');

    function openModal() {
        lightboxModal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    function closeModal() {
        lightboxModal.classList.remove('active');
        document.body.style.overflow = 'auto';
    }

    if (plotWrapper) plotWrapper.addEventListener('click', openModal);
    if (viewBtn) viewBtn.addEventListener('click', openModal);
    if (closeBtn) closeBtn.addEventListener('click', closeModal);

    if (lightboxModal) {
        lightboxModal.addEventListener('click', (e) => {
            if (e.target === lightboxModal) closeModal();
        });
    }

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && lightboxModal.classList.contains('active')) {
            closeModal();
        }
    });
}

// Sample Data Matrix Records (Generated from multi_disease_latent_aligned_cohort.csv)
const sampleMatrixData = []; // Record examples remain local.

let filteredData = [...sampleMatrixData];

function initSampleMatrixTable() {
    renderTable();
}

function renderTable() {
    const tbody = document.getElementById('matrixTbody');
    if (!tbody) return;

    tbody.innerHTML = '';
    
    if (filteredData.length === 0) {
        tbody.innerHTML = `<tr><td colspan="13" style="text-align: center; color: var(--text-muted); padding: 24px;">No cohort records match the search filter criteria.</td></tr>`;
        return;
    }

    filteredData.forEach(row => {
        const tr = document.createElement('tr');
        
        let diseaseClass = 'heart-theme';
        if (row.disease === 'Diabetes') diseaseClass = 'diabetes-theme';
        if (row.disease === 'Hypertension') diseaseClass = 'hypertension-theme';
        if (row.disease === 'Respiratory Disease') diseaseClass = 'respiratory-theme';

        tr.innerHTML = `
            <td>#${row.id}</td>
            <td><strong class="${diseaseClass}">${row.disease}</strong></td>
            <td>${row.age} yrs</td>
            <td>${row.sex}</td>
            <td>${row.sys_bp} / ${row.dia_bp}</td>
            <td>${row.bmi}</td>
            <td>${row.risk}</td>
            <td><code style="color:#00f2fe">${row.z1.toFixed(3)}</code></td>
            <td><code style="color:#00f2fe">${row.z2.toFixed(3)}</code></td>
            <td><code style="color:#00f2fe">${row.z3.toFixed(3)}</code></td>
            <td><code style="color:#00f2fe">${row.z4.toFixed(3)}</code></td>
            <td><code style="color:#00f2fe">${row.z5.toFixed(3)}</code></td>
            <td><code style="color:#00f2fe">${row.z6.toFixed(3)}</code></td>
        `;
        tbody.appendChild(tr);
    });
}

function initInteractiveFilters() {
    const searchInput = document.getElementById('tableSearchInput');
    const cohortFilter = document.getElementById('cohortFilterSelect');

    function applyFilters() {
        const query = searchInput ? searchInput.value.toLowerCase().trim() : '';
        const selectedCohort = cohortFilter ? cohortFilter.value : 'ALL';

        filteredData = sampleMatrixData.filter(row => {
            const matchesCohort = (selectedCohort === 'ALL') || (row.disease === selectedCohort);
            const matchesQuery = !query || 
                row.disease.toLowerCase().includes(query) || 
                row.age.toString().includes(query) ||
                row.sex.toLowerCase().includes(query);
            return matchesCohort && matchesQuery;
        });

        renderTable();
    }

    if (searchInput) searchInput.addEventListener('input', applyFilters);
    if (cohortFilter) cohortFilter.addEventListener('change', applyFilters);
}

// Download Aligned Matrix Trigger
function downloadAlignedCSV() {
    alert("Initiating download for 'multi_disease_latent_aligned_cohort.csv' (60,000 Patient Records matrix).");
    window.location.href = '../output/multi_disease_latent_aligned_cohort.csv';
}
