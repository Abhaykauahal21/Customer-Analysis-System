# UI Mockup Description

## Design Overview

The Customer Analysis System features a modern, clean interface with a professional color scheme, smooth animations, and intuitive navigation.

## Color Palette

### Light Theme
- **Primary**: Sky Blue (#0ea5e9)
- **Background**: Light Gray (#f9fafb)
- **Cards**: White (#ffffff)
- **Text**: Dark Gray (#1f2937)
- **Borders**: Light Gray (#e5e7eb)

### Dark Theme
- **Primary**: Sky Blue (#0ea5e9)
- **Background**: Dark Gray (#111827)
- **Cards**: Dark Gray (#1f2937)
- **Text**: Light Gray (#f9fafb)
- **Borders**: Dark Gray (#374151)

## Layout Structure

### Navigation Bar (Top)
- **Height**: 64px
- **Background**: White (light) / Dark Gray (dark)
- **Content**: 
  - Left: "📊 Customer Analysis System" (logo/title)
  - Right: Dark/Light theme toggle button (🌙/☀️)
- **Shadow**: Subtle bottom border with shadow

### Sidebar (Left)
- **Width**: 256px (desktop), collapsible on mobile
- **Background**: White (light) / Dark Gray (dark)
- **Menu Items**:
  - 📈 Dashboard
  - 🎯 Segmentation
  - 🔮 Prediction
  - 📊 Results
- **Active State**: Primary blue background with white text
- **Hover State**: Light gray background

### Main Content Area
- **Padding**: 24px (desktop), 16px (mobile)
- **Background**: Light gray (light) / Dark gray (dark)
- **Responsive**: Grid layout adapts to screen size

## Page Descriptions

### 1. Dashboard Page

**Layout:**
- Hero section with "Model Performance Dashboard" title
- Best Model card (gradient blue background, white text)
  - Shows model name, accuracy, precision, recall, F1-score
- Model Comparison Chart (large bar chart)
  - X-axis: Model names
  - Y-axis: Performance metrics (0-1 scale)
  - Multiple colored bars (blue, green, orange, purple)
- Model Cards Grid (4 columns on desktop, 2 on tablet, 1 on mobile)
  - Each card shows individual model metrics
  - White/dark gray background with shadow
  - Hover effect: Shadow increases

**Visual Elements:**
- Cards have rounded corners (12px)
- Subtle shadows for depth
- Smooth hover transitions
- Loading skeletons while data loads

### 2. Segmentation Page

**Layout:**
- Title: "Customer Segmentation (PCA + K-Means)"
- Info Cards Row (2 columns)
  - PCA Explained Variance card
  - Cluster Statistics card
- Large Scatter Plot (600px height)
  - X-axis: Principal Component 1
  - Y-axis: Principal Component 2
  - Multiple colored clusters (blue, green, orange, red, purple, pink)
  - Cluster centers marked with stars
  - Interactive tooltips on hover

**Visual Elements:**
- Color-coded clusters
- Legend showing cluster names
- Grid lines for readability
- Responsive chart sizing

### 3. Prediction Page

**Layout:**
- Title: "Purchase Likelihood Prediction"
- Large form with grouped sections:
  - Basic Information (Age, Income, etc.)
  - Spending Patterns (MntWines, MntFruits, etc.)
  - Purchase Behavior (NumWebPurchases, etc.)
  - Additional Information (Encoded features)
- Form fields in 3-column grid (desktop)
- Submit button (primary blue, large, right-aligned)

**Visual Elements:**
- Input fields with rounded borders
- Focus states: Blue ring around input
- Labels above each input
- Helper text for encoded features
- Error messages in red (if validation fails)

### 4. Results Page

**Layout:**
- Title: "Prediction Results"
- Results Grid (2x2 on desktop, 1 column on mobile):
  - **Prediction Card**: Large colored badge (green/red)
    - Shows "Likely to Purchase" or "Unlikely to Purchase"
    - Confidence bar below
  - **Probability Breakdown Card**: Two progress bars
    - Purchase probability (green)
    - No Purchase probability (red)
  - **Customer Segment Card**: Cluster number display
    - Large number in primary color
    - Distance from center
  - **PCA Coordinates Card**: Numerical values
    - PC1 and PC2 coordinates
- Action Buttons (centered, below cards)
  - "Make Another Prediction" (primary blue)
  - "View Segmentation" (gray)

**Visual Elements:**
- Color-coded prediction (green = purchase, red = no purchase)
- Progress bars with smooth animations
- Card shadows for depth
- Responsive grid layout

## Component Details

### Cards
- **Border Radius**: 12px (xl)
- **Padding**: 24px
- **Shadow**: Medium shadow (lg)
- **Hover**: Shadow increases slightly
- **Background**: White (light) / Dark gray (dark)

### Buttons
- **Primary**: Blue background, white text, rounded (lg)
- **Hover**: Slightly darker blue
- **Focus**: Blue ring outline
- **Disabled**: Reduced opacity, no pointer

### Input Fields
- **Border**: 1px solid gray
- **Border Radius**: 8px (lg)
- **Padding**: 8px 16px
- **Focus**: 2px blue ring
- **Background**: White (light) / Dark gray (dark)

### Charts
- **Background**: Transparent (inherits card background)
- **Grid Lines**: Light gray, dashed
- **Colors**: Distinct colors for each series
- **Tooltips**: Dark background, white text
- **Responsive**: Scales with container

## Responsive Breakpoints

- **Mobile**: < 640px (1 column layouts)
- **Tablet**: 640px - 1024px (2 column layouts)
- **Desktop**: > 1024px (3-4 column layouts)

## Animations

- **Page Transitions**: Smooth fade-in
- **Loading States**: Pulse animation on skeletons
- **Hover Effects**: Subtle scale/shadow increase
- **Button Clicks**: Brief scale-down effect
- **Chart Animations**: Smooth data entry animations

## Accessibility Features

- **Color Contrast**: WCAG AA compliant
- **Keyboard Navigation**: Full keyboard support
- **Screen Readers**: Semantic HTML, ARIA labels
- **Focus Indicators**: Visible focus rings
- **Alt Text**: Descriptive text for icons

## Typography

- **Headings**: Bold, larger sizes (2xl, 3xl)
- **Body**: Regular weight, readable sizes
- **Labels**: Medium weight, smaller sizes
- **Font Family**: System fonts (San Francisco, Segoe UI, etc.)

---

**Note**: This is a text-based description. For actual visual mockups, refer to the running application or create mockups using design tools like Figma, Sketch, or Adobe XD.

