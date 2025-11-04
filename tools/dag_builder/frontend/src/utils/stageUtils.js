// Utility functions for dynamic stage handling

/**
 * Generate a consistent color for a stage based on its name hash
 * This ensures the same stage always gets the same color
 */
export function getStageColor(stageName) {
  // Simple hash function for consistent colors
  let hash = 0;
  for (let i = 0; i < stageName.length; i++) {
    const char = stageName.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }

  // Convert hash to HSL color space for better color distribution
  const hue = Math.abs(hash) % 360;
  const saturation = 60 + (Math.abs(hash >> 8) % 25); // 60-85%
  const lightness = 45 + (Math.abs(hash >> 16) % 20); // 45-65%

  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

/**
 * Get a neutral icon for stages - no hardcoded logic
 */
export function getStageIcon() {
  // Return a generic, professional icon
  return '⚙️';
}

/**
 * Generate a unique node ID
 */
export function generateNodeId() {
  return `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Generate a unique edge ID
 */
export function generateEdgeId() {
  return `edge_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Validate if a connection is valid between two stages
 */
export function validateConnection(sourceStage, targetStage, inputName) {
  // Check if the target stage accepts the specified input
  const mandatoryInputs = targetStage.mandatory_inputs || [];
  const optionalInputs = targetStage.optional_inputs || [];

  return mandatoryInputs.includes(inputName) || optionalInputs.includes(inputName);
}

/**
 * Get connection suggestions for a stage
 */
export function getConnectionSuggestions(stage, availableStages) {
  const suggestions = [];
  const mandatoryInputs = stage.mandatory_inputs || [];
  const optionalInputs = stage.optional_inputs || [];

  // For each input, suggest compatible stages
  [...mandatoryInputs, ...optionalInputs].forEach(inputName => {
    availableStages.forEach(otherStage => {
      if (otherStage.name !== stage.name) {
        suggestions.push({
          inputName,
          suggestedStage: otherStage.name,
          stage: otherStage
        });
      }
    });
  });

  return suggestions;
}

/**
 * Format stage name for display (remove common suffixes)
 */
export function formatStageName(stageName) {
  // Remove common class suffixes for cleaner display
  return stageName
    .replace(/Stage$/, '')
    .replace(/Executor$/, '')
    .replace(/([A-Z])/g, ' $1')
    .trim();
}

/**
 * Get input type color based on whether it's mandatory or optional
 */
export function getInputTypeColor(isMandatory) {
  return isMandatory ? '#dc3545' : '#6c757d';
}

/**
 * Get input type label
 */
export function getInputTypeLabel(isMandatory) {
  return isMandatory ? 'required' : 'optional';
}

/**
 * Generate a unique stage name by appending a counter if needed
 * @param {string} baseName - The original stage name
 * @param {Array} existingNodes - Array of existing nodes to check against
 * @returns {string} - Unique stage name
 */
export function generateUniqueStageName(baseName, existingNodes) {
  if (!existingNodes || existingNodes.length === 0) {
    return baseName;
  }

  // Get all existing stage names (use the actual name field)
  const existingNames = existingNodes.map(node => {
    return node.data?.name;
  }).filter(Boolean);

  // If the base name is unique, return it
  if (!existingNames.includes(baseName)) {
    return baseName;
  }

  // Find the highest counter for this base name
  let counter = 1;
  let uniqueName = `${baseName}_${counter}`;

  while (existingNames.includes(uniqueName)) {
    counter++;
    uniqueName = `${baseName}_${counter}`;
  }

  return uniqueName;
}

/**
 * Check if a stage name is unique among existing nodes
 * @param {string} name - The name to check
 * @param {Array} existingNodes - Array of existing nodes to check against
 * @param {string} excludeNodeId - Optional node ID to exclude from the check (for editing)
 * @returns {boolean} - True if the name is unique
 */
export function isStageNameUnique(name, existingNodes, excludeNodeId = null) {
  if (!existingNodes || existingNodes.length === 0) {
    return true;
  }

  const existingNames = existingNodes
    .filter(node => node.id !== excludeNodeId) // Exclude the current node if editing
    .map(node => {
      const customName = node.data?.customName;
      const originalName = node.data?.name;
      return customName || originalName;
    })
    .filter(Boolean);

  return !existingNames.includes(name);
}

/**
 * Get the display name for a stage (custom name or original name)
 * @param {Object} nodeData - The node data object
 * @returns {string} - The display name
 */
export function getStageDisplayName(nodeData) {
  return nodeData?.customName || nodeData?.name || 'Untitled Stage';
}
