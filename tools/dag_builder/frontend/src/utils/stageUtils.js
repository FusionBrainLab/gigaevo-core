// Utility functions for dynamic stage handling

/**
 * Generate a neutral gray color for minimal, professional design
 * All stages use the same calm gray palette
 */
export function getStageColor(stageName) {
  // Minimal design: neutral gray for all stages
  // Focus on structure, not decoration
  return 'hsl(220, 9%, 46%)'; // Subtle blue-gray
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

/**
 * Extract the inner type from Optional[Type] or Union[Type, None]
 * @param {string} typeName - The type name (e.g., "Optional[ProgramIO]")
 * @returns {string} - The inner type (e.g., "ProgramIO") or original if not optional
 */
export function extractInnerType(typeName) {
  if (!typeName) return typeName;

  // Handle Optional[Type]
  const optionalMatch = typeName.match(/^Optional\[(.+)\]$/);
  if (optionalMatch) {
    return optionalMatch[1];
  }

  // Handle Union[Type, None] or Union[None, Type]
  const unionMatch = typeName.match(/^Union\[(.+),\s*None\]$|^Union\[None,\s*(.+)\]$/);
  if (unionMatch) {
    return unionMatch[1] || unionMatch[2];
  }

  return typeName;
}

/**
 * Generate a consistent color for a given type
 * Same type always gets the same color for easy visual matching
 * @param {string} typeName - The type name (e.g., "ProgramIO", "str", "Optional[Dict[str, Any]]")
 * @returns {string} - HSL color string
 */
export function getTypeColor(typeName) {
  if (!typeName) {
    return '#6c757d'; // Default gray for undefined types
  }

  // Extract inner type for Optional/Union types to ensure consistent colors
  const baseType = extractInnerType(typeName);

  // Predefined palette of distinct, accessible colors
  const colorPalette = [
    'hsl(210, 79%, 46%)',  // Blue
    'hsl(340, 82%, 52%)',  // Pink
    'hsl(291, 64%, 42%)',  // Purple
    'hsl(262, 52%, 47%)',  // Deep purple
    'hsl(39, 100%, 50%)',  // Orange
    'hsl(122, 39%, 49%)',  // Green
    'hsl(187, 71%, 42%)',  // Teal
    'hsl(14, 100%, 57%)',  // Red-orange
    'hsl(45, 100%, 51%)',  // Yellow
    'hsl(171, 100%, 37%)', // Cyan
    'hsl(4, 90%, 58%)',    // Red
    'hsl(24, 100%, 50%)',  // Deep orange
  ];

  // Simple hash function for consistent color assignment
  let hash = 0;
  for (let i = 0; i < baseType.length; i++) {
    hash = ((hash << 5) - hash) + baseType.charCodeAt(i);
    hash = hash & hash; // Convert to 32-bit integer
  }

  // Map hash to palette index
  const index = Math.abs(hash) % colorPalette.length;
  return colorPalette[index];
}
